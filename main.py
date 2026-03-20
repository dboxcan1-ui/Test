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
import fal_client
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from dotenv import load_dotenv

load_dotenv()

# Strip any trailing whitespace/newlines from FAL_KEY (common copy-paste issue)
if os.getenv("FAL_KEY"):
    os.environ["FAL_KEY"] = os.environ["FAL_KEY"].strip()

app = FastAPI(title="ref2vid")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FAL_ENDPOINT = "fal-ai/kling-video/o1/standard/reference-to-video"
REFS_DIR = Path("refs_store")
REFS_DIR.mkdir(exist_ok=True)


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB (Kling limit is 10 MB)


def compress_image(content: bytes, suffix: str) -> tuple[bytes, str]:
    """Resize/recompress image to fit under MAX_UPLOAD_BYTES."""
    if len(content) <= MAX_UPLOAD_BYTES:
        return content, suffix
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


# ── Static pages ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html") as f:
        return f.read()


# ── Reference library ─────────────────────────────────────────────────────────

@app.get("/references")
async def list_references():
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
    return {"ok": True}


@app.get("/references/{ref_id}/images/{filename}")
async def get_reference_image(ref_id: str, filename: str):
    # Prevent path traversal
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
):
    if not os.getenv("FAL_KEY"):
        async def err():
            yield sse({"status": "error", "message": "FAL_KEY not set. Add it to your .env file."})
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

    # cfg_scale controls how strictly Kling follows the text prompt.
    # Keep it moderate-to-high so the prompt is actually respected.
    cfg_scale = max(0.4, min(0.7, 0.4 + motion_strength * 0.3))

    # Kling O1 needs @Element1 in the prompt to anchor the reference character.
    full_prompt = prompt if "@Element1" in prompt else f"@Element1 {prompt}"

    async def event_stream():
        try:
            # Only upload the top-weighted image — Vidu uses one reference per subject.
            yield sse({"status": "uploading", "message": "Uploading reference image to fal storage…"})
            primary_img = sorted_images[0]
            suffix = os.path.splitext(primary_img.filename or ".jpg")[1] or ".jpg"
            content = await primary_img.read()
            primary_url = await upload_image(content, suffix)

            yield sse({"status": "submitted", "message": "Image uploaded. Submitting to fal.ai…", "progress": 20})

            # One element = one character. Passing multiple images caused
            # Kling to render each as a separate person in the scene.
            arguments = {
                "prompt": full_prompt,
                "elements": [{"frontal_image_url": primary_url}],
                "aspect_ratio": aspect_ratio,
                "cfg_scale": cfg_scale,
                "enable_safety_checker": False,
            }

            result = None
            for attempt in range(3):
                try:
                    handle = await fal_client.submit_async(FAL_ENDPOINT, arguments=arguments)
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
                    if "downstream_service_error" in str(e) and attempt < 2:
                        wait = 2 ** attempt
                        yield sse({"status": "queued", "message": f"Downstream error, retrying in {wait}s… (attempt {attempt+2}/3)", "progress": 20})
                        await asyncio.sleep(wait)
                    else:
                        raise
            if result is None:
                raise RuntimeError("fal.ai downstream service failed after 3 attempts")

            video = result.get("video") or (result.get("videos") or [None])[0]
            if not video:
                yield sse({"status": "error", "message": f"Unexpected response shape: {list(result.keys())}"})
                return

            video_url = video.get("url") if isinstance(video, dict) else str(video)
            yield sse({
                "status": "complete",
                "message": "Video ready!",
                "video_url": video_url,
                "prompt_used": full_prompt,
                "progress": 100,
            })

        except Exception as exc:
            yield sse({"status": "error", "message": str(exc), "progress": 0})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
