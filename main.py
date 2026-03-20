import os
import json
import asyncio
import tempfile
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import io
import modal
import fal_client
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from dotenv import load_dotenv

load_dotenv()

# Strip any trailing whitespace/newlines from keys (common copy-paste issue)
if os.getenv("FAL_KEY"):
    os.environ["FAL_KEY"] = os.environ["FAL_KEY"].strip()
if os.getenv("MODAL_TOKEN_ID"):
    os.environ["MODAL_TOKEN_ID"] = os.environ["MODAL_TOKEN_ID"].strip()
if os.getenv("MODAL_TOKEN_SECRET"):
    os.environ["MODAL_TOKEN_SECRET"] = os.environ["MODAL_TOKEN_SECRET"].strip()

app = FastAPI(title="ref2vid")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Storage paths ──────────────────────────────────────────────────────────────
# DATA_DIR is injected as "/data" when running inside the Modal web endpoint.
# Locally it falls back to the legacy directories so nothing breaks.
_DATA_DIR_ENV = os.getenv("DATA_DIR")
if _DATA_DIR_ENV:
    DATA_DIR    = Path(_DATA_DIR_ENV)
    REFS_DIR    = DATA_DIR / "refs"
    VIDEOS_DIR  = DATA_DIR / "videos"
    PROMPTS_FILE = DATA_DIR / "prompts.json"
    LOGS_DIR    = DATA_DIR / "logs"
else:
    DATA_DIR    = None
    REFS_DIR    = Path("refs_store")
    VIDEOS_DIR  = Path("/tmp/videos")
    PROMPTS_FILE = Path("prompts.json")
    LOGS_DIR    = None  # skip logs when running locally without a volume

REFS_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
if LOGS_DIR:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Modal volume helpers ───────────────────────────────────────────────────────
# _volume is set by modal_wan.py's web() entry-point when running inside Modal.
_volume = None


def _reload_volume() -> None:
    if _volume is not None:
        _volume.reload()


def _commit_volume() -> None:
    if _volume is not None:
        _volume.commit()


# ── True when running inside a Modal container (no explicit token needed) ─────
_inside_modal = bool(os.getenv("MODAL_TASK_ID"))


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB (Kling limit is 10 MB)


def compress_image(content: bytes, suffix: str) -> tuple[bytes, str]:
    """Convert to JPEG and resize/recompress to fit under MAX_UPLOAD_BYTES."""
    img = Image.open(io.BytesIO(content)).convert("RGB")
    quality = 85
    scale = 1.0
    while True:
        w, h = int(img.width * scale), int(img.height * scale)
        resized = img.resize((w, h), Image.LANCZOS) if scale < 1.0 else img
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()
        if len(data) <= MAX_UPLOAD_BYTES:
            return data, ".jpg"
        if quality > 60:
            quality -= 10
        else:
            scale *= 0.8


async def upload_image(content: bytes, suffix: str) -> str:
    content, suffix = await asyncio.to_thread(compress_image, content, suffix)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(content)
        tmp_path = f.name
    try:
        url = await asyncio.to_thread(fal_client.upload_file, tmp_path)
        return url
    finally:
        os.unlink(tmp_path)


# ── Prompt history ─────────────────────────────────────────────────────────────

def _save_prompt(text: str) -> None:
    """Prepend to recent-prompts list (max 3, deduplicated)."""
    if not text.strip():
        return
    try:
        prompts = json.loads(PROMPTS_FILE.read_text()) if PROMPTS_FILE.exists() else []
        prompts = [p for p in prompts if p.get("text") != text]
        prompts.insert(0, {
            "text": text,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        PROMPTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        PROMPTS_FILE.write_text(json.dumps(prompts[:3], indent=2))
        asyncio.get_event_loop().run_in_executor(None, _commit_volume)
    except Exception:
        pass


# ── Generation log ─────────────────────────────────────────────────────────────

def _write_log(entry: dict) -> None:
    if LOGS_DIR is None:
        return
    try:
        ts  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        vid = entry.get("video_id", "err")
        (LOGS_DIR / f"{ts}_{vid}.json").write_text(json.dumps(entry, indent=2))
        _commit_volume()
    except Exception:
        pass


# ── Static pages ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return (Path(__file__).parent / "index.html").read_text()


# ── Prompts endpoint ───────────────────────────────────────────────────────────

@app.get("/prompts")
async def get_prompts():
    await asyncio.to_thread(_reload_volume)
    if not PROMPTS_FILE.exists():
        return []
    try:
        return json.loads(PROMPTS_FILE.read_text())
    except Exception:
        return []


# ── Reference library ─────────────────────────────────────────────────────────

@app.get("/references")
async def list_references():
    await asyncio.to_thread(_reload_volume)
    refs = []
    for d in sorted(REFS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        meta_file = d / "meta.json"
        if d.is_dir() and meta_file.exists():
            refs.append(json.loads(meta_file.read_text()))
    return refs


@app.post("/references")
async def save_reference(
    name: str = Form(...),
    images: List[UploadFile] = File(...),
    weights: str = Form(default="[]"),
    prompt: str = Form(default=""),
):
    if not name.strip():
        raise HTTPException(400, "Name is required")
    if len(images) < 2 or len(images) > 6:
        raise HTTPException(400, f"Expected 2–6 images, got {len(images)}")

    ref_id = uuid.uuid4().hex[:10]
    ref_dir = REFS_DIR / ref_id
    ref_dir.mkdir()

    filenames = []
    for i, img in enumerate(images):
        suffix = Path(img.filename or "img.jpg").suffix or ".jpg"
        fname = f"img_{i}{suffix}"
        content = await img.read()
        (ref_dir / fname).write_bytes(content)
        filenames.append(fname)

    try:
        parsed_weights = json.loads(weights)
    except Exception:
        parsed_weights = [1.0] * len(images)

    meta = {
        "id": ref_id,
        "name": name.strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "weights": parsed_weights,
        "filenames": filenames,
        "prompt": prompt,
    }
    (ref_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    await asyncio.to_thread(_commit_volume)
    return meta


@app.get("/references/{ref_id}")
async def get_reference(ref_id: str):
    meta_file = REFS_DIR / ref_id / "meta.json"
    if not meta_file.exists():
        raise HTTPException(404, "Reference set not found")
    return json.loads(meta_file.read_text())


@app.delete("/references/{ref_id}")
async def delete_reference(ref_id: str):
    ref_dir = REFS_DIR / ref_id
    if not ref_dir.exists():
        raise HTTPException(404, "Reference set not found")
    shutil.rmtree(ref_dir)
    await asyncio.to_thread(_commit_volume)
    return {"ok": True}


@app.get("/references/{ref_id}/images/{filename}")
async def get_reference_image(ref_id: str, filename: str):
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(400, "Invalid filename")
    path = REFS_DIR / ref_id / filename
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


# ── Video generation ──────────────────────────────────────────────────────────

@app.post("/generate")
async def generate(
    images: List[UploadFile] = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(default="low resolution, blurry, worst quality, artifacts"),
    image_weights: str = Form(default="[]"),
    aspect_ratio: str = Form(default="9:16"),
    resolution: str = Form(default="720p"),
    duration: str = Form(default="5"),
    motion_strength: float = Form(default=0.5),
    model: str = Form(default="wan"),
):
    use_kling = model == "kling"

    if use_kling and not os.getenv("FAL_KEY"):
        async def err():
            yield sse({"status": "error", "message": "FAL_KEY not set. Required for Kling model."})
        return StreamingResponse(err(), media_type="text/event-stream")

    if not use_kling and not os.getenv("MODAL_TOKEN_ID") and not _inside_modal:
        async def err():
            yield sse({"status": "error", "message": "MODAL_TOKEN_ID not set. Required for WAN model."})
        return StreamingResponse(err(), media_type="text/event-stream")

    if len(images) < 1 or len(images) > 6:
        async def err():
            yield sse({"status": "error", "message": f"Expected 1–6 images, got {len(images)}."})
        return StreamingResponse(err(), media_type="text/event-stream")

    try:
        weights = json.loads(image_weights)
    except Exception:
        weights = [1.0] * len(images)
    while len(weights) < len(images):
        weights.append(1.0)

    image_pairs = sorted(zip(weights, images), key=lambda x: x[0], reverse=True)
    sorted_weights, sorted_images = zip(*image_pairs)

    if motion_strength < 0.3:
        motion_tag = " Subtle, gentle motion."
    elif motion_strength < 0.6:
        motion_tag = " Smooth, natural motion."
    elif motion_strength < 0.85:
        motion_tag = " Dynamic, expressive motion."
    else:
        motion_tag = " Highly dynamic, energetic motion."

    if use_kling:
        full_prompt = prompt if "@Element1" in prompt else f"@Element1 {prompt}"
        cfg_scale = round(max(0.5, min(1.0, 0.5 + motion_strength * 0.5)), 2)
    else:
        full_prompt = prompt if "Character1" in prompt else f"Character1 {prompt}"

    gen_start = datetime.now(timezone.utc)

    async def event_stream():
        try:
            primary_img = sorted_images[0]
            suffix = os.path.splitext(primary_img.filename or ".jpg")[1] or ".jpg"
            content = await primary_img.read()

            if use_kling:
                # ── Kling via fal.ai ──────────────────────────────────────────
                yield sse({"status": "uploading", "message": "Uploading image to fal storage…"})
                primary_url = await upload_image(content, suffix)
                yield sse({"status": "submitted", "message": "Image uploaded. Submitting to fal.ai…", "progress": 20})

                fal_endpoint = "fal-ai/kling-video/v1.6/standard/elements"
                arguments = {
                    "prompt": full_prompt,
                    "negative_prompt": negative_prompt,
                    "input_image_urls": [primary_url],
                    "elements": [{"frontal_image_url": primary_url, "reference_image_urls": [primary_url]}],
                    "aspect_ratio": aspect_ratio,
                    "duration": str(duration),
                    "cfg_scale": cfg_scale,
                    "enable_safety_checker": False,
                }

                result = None
                for attempt in range(3):
                    try:
                        handle = await fal_client.submit_async(fal_endpoint, arguments=arguments)
                        yield sse({"status": "queued", "message": "Job queued. Waiting for a worker…", "request_id": handle.request_id, "progress": 25})

                        logs_seen = 0
                        async for event in handle.iter_events(with_logs=True):
                            if isinstance(event, fal_client.Queued):
                                yield sse({
                                    "status": "queued",
                                    "message": f"Position in queue: {event.position}",
                                    "position": event.position,
                                    "progress": 25,
                                })
                            elif isinstance(event, (fal_client.InProgress, fal_client.Completed)):
                                new_logs = event.logs[logs_seen:]
                                logs_seen = len(event.logs)
                                for log in new_logs:
                                    yield sse({"status": "processing", "message": log.get("message", ""), "progress": -1})

                        result = await handle.get()
                        break
                    except Exception as e:
                        err_str = str(e)
                        if "downstream_service_error" in err_str and attempt < 2:
                            wait = 2 ** attempt
                            yield sse({"status": "queued", "message": f"Model error, retrying in {wait}s… (attempt {attempt+2}/3)", "progress": 20})
                            await asyncio.sleep(wait)
                        else:
                            raise

                if result is None:
                    raise RuntimeError(
                        "The model failed after 3 attempts (downstream_service_error). "
                        "Try a different prompt or image."
                    )

                video = result.get("video") or (result.get("videos") or [None])[0]
                if not video:
                    yield sse({"status": "error", "message": f"Unexpected response shape: {list(result.keys())}"})
                    return

                video_url = video.get("url") if isinstance(video, dict) else str(video)

                _save_prompt(prompt)
                elapsed = (datetime.now(timezone.utc) - gen_start).total_seconds()
                _write_log({
                    "model": "kling", "prompt": full_prompt, "aspect_ratio": aspect_ratio,
                    "duration": duration, "status": "complete", "video_url": video_url,
                    "created_at": gen_start.isoformat(), "elapsed_seconds": round(elapsed, 1),
                })

                yield sse({
                    "status": "complete",
                    "message": "Video ready!",
                    "video_url": video_url,
                    "prompt_used": full_prompt,
                    "progress": 100,
                })

            else:
                # ── WAN via Modal ─────────────────────────────────────────────
                yield sse({"status": "uploading", "message": "Preparing image…"})
                compressed, _ = await asyncio.to_thread(compress_image, content, suffix)

                yield sse({"status": "submitted", "message": "Submitting to Modal — container may need a moment to start…", "progress": 15})

                WanGenerator = modal.Cls.from_name("ref2vid-wan", "WanGenerator")

                # Run the blocking .remote() call in a thread so the event loop
                # can keep sending SSE heartbeats while the Modal container
                # cold-starts and runs inference (can take several minutes).
                loop = asyncio.get_event_loop()
                result_future: asyncio.Future = loop.run_in_executor(
                    None,
                    lambda: WanGenerator().generate.remote(
                        compressed,
                        full_prompt,
                        negative_prompt,
                        aspect_ratio,
                        int(duration),
                    ),
                )

                # Heartbeat every 20 s so the browser connection stays alive.
                _HEARTBEAT_S = 20
                while not result_future.done():
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(result_future), timeout=_HEARTBEAT_S
                        )
                    except asyncio.TimeoutError:
                        yield sse({
                            "status":  "submitted",
                            "message": "Container starting / running inference — please wait…",
                            "progress": 15,
                        })

                video_id = result_future.result()  # raises if the remote call failed
                if not video_id:
                    raise RuntimeError("No video returned from GPU container.")

                # Sync the data volume so we can serve the file immediately
                await asyncio.to_thread(_reload_volume)

                _save_prompt(prompt)
                elapsed = (datetime.now(timezone.utc) - gen_start).total_seconds()
                _write_log({
                    "model": "wan", "prompt": full_prompt, "aspect_ratio": aspect_ratio,
                    "duration": duration, "status": "complete", "video_id": video_id,
                    "created_at": gen_start.isoformat(), "elapsed_seconds": round(elapsed, 1),
                })

                yield sse({
                    "status": "complete",
                    "message": "Video ready!",
                    "video_url": f"/video/{video_id}",
                    "prompt_used": full_prompt,
                    "progress": 100,
                })

        except Exception as exc:
            msg = str(exc)
            if "downstream_service_error" in msg:
                msg = "Model error: fal.ai downstream service failed. Try again or use a different image."
            _write_log({
                "model": model, "prompt": full_prompt, "status": "error", "error": msg,
                "created_at": gen_start.isoformat(),
            })
            yield sse({"status": "error", "message": msg, "progress": 0})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/video/{video_id}")
async def serve_video(video_id: str):
    if len(video_id) != 32 or not all(c in "0123456789abcdef" for c in video_id):
        raise HTTPException(400, "Invalid video ID")
    path = VIDEOS_DIR / f"{video_id}.mp4"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="video/mp4")
