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
from fastapi.responses import HTMLResponse, FileResponse
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
_DATA_DIR_ENV = os.getenv("DATA_DIR")
if _DATA_DIR_ENV:
    DATA_DIR      = Path(_DATA_DIR_ENV)
    REFS_DIR      = DATA_DIR / "refs"
    VIDEOS_DIR    = DATA_DIR / "videos"
    PROMPTS_FILE  = DATA_DIR / "prompts.json"
    LOGS_DIR      = DATA_DIR / "logs"
    LAST_GEN_FILE = DATA_DIR / "last_gen.json"
    JOBS_DIR      = DATA_DIR / "jobs"
else:
    DATA_DIR      = None
    REFS_DIR      = Path("refs_store")
    VIDEOS_DIR    = Path("/tmp/videos")
    PROMPTS_FILE  = Path("prompts.json")
    LOGS_DIR      = None
    LAST_GEN_FILE = Path("last_gen.json")
    JOBS_DIR      = Path("jobs_store")

for _d in (REFS_DIR, VIDEOS_DIR, JOBS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
if LOGS_DIR:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Modal volume helpers ───────────────────────────────────────────────────────
_volume = None


def _reload_volume() -> None:
    if _volume is not None:
        _volume.reload()


def _commit_volume() -> None:
    if _volume is not None:
        _volume.commit()


# ── True when running inside a Modal container ────────────────────────────────
_inside_modal = bool(os.getenv("MODAL_TASK_ID"))

# ── Image helpers ─────────────────────────────────────────────────────────────
MAX_UPLOAD_BYTES = 8 * 1024 * 1024


def compress_image(content: bytes, suffix: str) -> tuple[bytes, str]:
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
    if not text.strip():
        return
    try:
        prompts = json.loads(PROMPTS_FILE.read_text()) if PROMPTS_FILE.exists() else []
        prompts = [p for p in prompts if p.get("text") != text]
        prompts.insert(0, {"text": text, "created_at": datetime.now(timezone.utc).isoformat()})
        PROMPTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        PROMPTS_FILE.write_text(json.dumps(prompts[:3], indent=2))
        _commit_volume()
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

# ── GPU warm status ────────────────────────────────────────────────────────────
SCALEDOWN_SECONDS = 600  # must match modal_wan.py scaledown_window


def _mark_last_gen() -> None:
    try:
        LAST_GEN_FILE.write_text(json.dumps({"at": datetime.now(timezone.utc).isoformat()}))
        _commit_volume()
    except Exception:
        pass

# ── Job queue ──────────────────────────────────────────────────────────────────
_jobs: dict[str, dict] = {}


def _save_job(job: dict) -> None:
    try:
        (JOBS_DIR / f"{job['id']}.json").write_text(json.dumps(job, indent=2))
        _commit_volume()
    except Exception:
        pass


def _upd(job_id: str, **kw) -> None:
    if job_id in _jobs:
        _jobs[job_id].update(kw)


async def _run_job(
    job_id: str,
    model: str,
    image_bytes: bytes,
    suffix: str,
    prompt: str,
    negative_prompt: str,
    full_prompt: str,
    aspect_ratio: str,
    duration: str,
    cfg_scale: float,
    gen_start: datetime,
) -> None:
    try:
        if model == "kling":
            _upd(job_id, message="Uploading image…", progress=10)
            primary_url = await upload_image(image_bytes, suffix)
            _upd(job_id, message="Submitted to fal.ai…", progress=20)

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
                    _upd(job_id, message="Job queued…", progress=25, request_id=handle.request_id)

                    logs_seen = 0
                    async for event in handle.iter_events(with_logs=True):
                        if isinstance(event, fal_client.Queued):
                            _upd(job_id, message=f"Queue position: {event.position}", progress=25)
                        elif isinstance(event, (fal_client.InProgress, fal_client.Completed)):
                            new_logs = event.logs[logs_seen:]
                            logs_seen = len(event.logs)
                            for log in new_logs:
                                if log.get("message"):
                                    _upd(job_id, message=log["message"], progress=50)

                    result = await handle.get()
                    break
                except Exception as e:
                    if "downstream_service_error" in str(e) and attempt < 2:
                        _upd(job_id, message=f"Retrying… (attempt {attempt+2}/3)", progress=20)
                        await asyncio.sleep(2 ** attempt)
                    else:
                        raise

            if result is None:
                raise RuntimeError("Model failed after 3 attempts (downstream_service_error).")

            video = result.get("video") or (result.get("videos") or [None])[0]
            if not video:
                raise RuntimeError(f"Unexpected response shape: {list(result.keys())}")
            video_url = video.get("url") if isinstance(video, dict) else str(video)

            _save_prompt(prompt)
            elapsed = (datetime.now(timezone.utc) - gen_start).total_seconds()
            _write_log({
                "model": "kling", "prompt": full_prompt, "aspect_ratio": aspect_ratio,
                "duration": duration, "status": "complete", "video_url": video_url,
                "created_at": gen_start.isoformat(), "elapsed_seconds": round(elapsed, 1),
            })
            _upd(job_id,
                 status="complete", message="Done!", progress=100,
                 video_url=video_url, prompt_used=full_prompt,
                 completed_at=datetime.now(timezone.utc).isoformat(),
                 elapsed=round(elapsed, 1))

        else:  # wan
            _upd(job_id, message="Preparing image…", progress=10)
            compressed, _ = await asyncio.to_thread(compress_image, image_bytes, suffix)
            _upd(job_id, message="Submitting to Modal — container may need a moment to start…", progress=15)

            WanGen = modal.Cls.from_name("ref2vid-wan", "WanGenerator")
            loop = asyncio.get_running_loop()
            result_future = loop.run_in_executor(
                None,
                lambda: WanGen().generate.remote(
                    compressed, full_prompt, negative_prompt, aspect_ratio, int(duration)
                ),
            )

            while not result_future.done():
                try:
                    await asyncio.wait_for(asyncio.shield(result_future), timeout=20)
                except asyncio.TimeoutError:
                    _upd(job_id, message="Container starting / running inference — please wait…", progress=15)

            video_id = result_future.result()
            if not video_id:
                raise RuntimeError("No video returned from GPU container.")

            await asyncio.to_thread(_reload_volume)
            _mark_last_gen()
            _save_prompt(prompt)
            elapsed = (datetime.now(timezone.utc) - gen_start).total_seconds()
            _write_log({
                "model": "wan", "prompt": full_prompt, "aspect_ratio": aspect_ratio,
                "duration": duration, "status": "complete", "video_id": video_id,
                "created_at": gen_start.isoformat(), "elapsed_seconds": round(elapsed, 1),
            })
            _upd(job_id,
                 status="complete", message="Done!", progress=100,
                 video_url=f"/video/{video_id}", prompt_used=full_prompt,
                 completed_at=datetime.now(timezone.utc).isoformat(),
                 elapsed=round(elapsed, 1))

    except Exception as exc:
        msg = str(exc)
        if "downstream_service_error" in msg:
            msg = "Model error: fal.ai downstream service failed. Try again or use a different image."
        _write_log({
            "model": model, "prompt": full_prompt, "status": "error", "error": msg,
            "created_at": gen_start.isoformat(),
        })
        _upd(job_id, status="error", message=msg, progress=0)
    finally:
        _save_job(_jobs[job_id])

# ── Static pages ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    return (Path(__file__).parent / "index.html").read_text()

# ── GPU warm status ────────────────────────────────────────────────────────────

@app.get("/gpu-status")
async def gpu_status():
    await asyncio.to_thread(_reload_volume)
    if not LAST_GEN_FILE.exists():
        return {"status": "unknown", "seconds_ago": None}
    try:
        data = json.loads(LAST_GEN_FILE.read_text())
        at = datetime.fromisoformat(data["at"])
        seconds_ago = int((datetime.now(timezone.utc) - at).total_seconds())
        if seconds_ago < SCALEDOWN_SECONDS * 0.8:
            status = "warm"
        elif seconds_ago < SCALEDOWN_SECONDS:
            status = "likely_warm"
        else:
            status = "cold"
        return {"status": status, "seconds_ago": seconds_ago}
    except Exception:
        return {"status": "unknown", "seconds_ago": None}

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

# ── Reference library ──────────────────────────────────────────────────────────

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
        (ref_dir / fname).write_bytes(await img.read())
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

# ── Video generation ───────────────────────────────────────────────────────────

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
        raise HTTPException(400, "FAL_KEY not set — required for Kling model.")
    if not use_kling and not os.getenv("MODAL_TOKEN_ID") and not _inside_modal:
        raise HTTPException(400, "MODAL_TOKEN_ID not set — required for WAN model.")
    if len(images) < 1 or len(images) > 6:
        raise HTTPException(400, f"Expected 1–6 images, got {len(images)}.")

    try:
        weights = json.loads(image_weights)
    except Exception:
        weights = [1.0] * len(images)
    while len(weights) < len(images):
        weights.append(1.0)

    image_pairs = sorted(zip(weights, images), key=lambda x: x[0], reverse=True)
    sorted_weights, sorted_images = zip(*image_pairs)

    primary_img = sorted_images[0]
    suffix = os.path.splitext(primary_img.filename or ".jpg")[1] or ".jpg"
    image_bytes = await primary_img.read()

    if use_kling:
        full_prompt = prompt if "@Element1" in prompt else f"@Element1 {prompt}"
        cfg_scale = round(max(0.5, min(1.0, 0.5 + motion_strength * 0.5)), 2)
    else:
        full_prompt = prompt if "Character1" in prompt else f"Character1 {prompt}"
        cfg_scale = 0.0

    job_id = uuid.uuid4().hex[:12]
    gen_start = datetime.now(timezone.utc)
    _jobs[job_id] = {
        "id": job_id,
        "status": "running",
        "message": "Starting…",
        "progress": 5,
        "model": model,
        "prompt": prompt,
        "prompt_used": full_prompt,
        "aspect_ratio": aspect_ratio,
        "duration": duration,
        "created_at": gen_start.isoformat(),
    }

    asyncio.create_task(_run_job(
        job_id=job_id, model=model, image_bytes=image_bytes, suffix=suffix,
        prompt=prompt, negative_prompt=negative_prompt, full_prompt=full_prompt,
        aspect_ratio=aspect_ratio, duration=duration, cfg_scale=cfg_scale,
        gen_start=gen_start,
    ))

    return {"job_id": job_id}

# ── Job status endpoints ───────────────────────────────────────────────────────

@app.get("/job/{job_id}")
async def get_job(job_id: str):
    if job_id in _jobs:
        return _jobs[job_id]
    # Fall back to persisted file (e.g. after container restart)
    job_file = JOBS_DIR / f"{job_id}.json"
    if job_file.exists():
        await asyncio.to_thread(_reload_volume)
        if job_file.exists():
            return json.loads(job_file.read_text())
    raise HTTPException(404, "Job not found")


@app.get("/jobs")
async def list_jobs():
    return sorted(_jobs.values(), key=lambda j: j["created_at"], reverse=True)[:20]

# ── Video serving ──────────────────────────────────────────────────────────────

@app.get("/video/{video_id}")
async def serve_video(video_id: str):
    if len(video_id) != 32 or not all(c in "0123456789abcdef" for c in video_id):
        raise HTTPException(400, "Invalid video ID")
    path = VIDEOS_DIR / f"{video_id}.mp4"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="video/mp4")
