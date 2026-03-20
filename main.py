import os
import json
import asyncio
import tempfile
from typing import List, Optional

import fal_client
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="ref2vid")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FAL_ENDPOINT = "wan/v2.6/reference-to-video/flash"


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def upload_image(content: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(content)
        tmp_path = f.name
    try:
        url = await asyncio.to_thread(fal_client.upload_file, tmp_path)
        return url
    finally:
        os.unlink(tmp_path)


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html") as f:
        return f.read()


@app.post("/generate")
async def generate(
    images: List[UploadFile] = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(default="low resolution, blurry, worst quality, artifacts"),
    image_weights: str = Form(default="[]"),
    aspect_ratio: str = Form(default="16:9"),
    resolution: str = Form(default="720p"),
    duration: str = Form(default="5"),
    motion_strength: float = Form(default=0.5),
):
    if not os.getenv("FAL_KEY"):
        async def err():
            yield sse({"status": "error", "message": "FAL_KEY not set. Add it to your .env file."})
        return StreamingResponse(err(), media_type="text/event-stream")

    if len(images) < 2 or len(images) > 6:
        async def err():
            yield sse({"status": "error", "message": f"Expected 2–6 images, got {len(images)}."})
        return StreamingResponse(err(), media_type="text/event-stream")

    # Parse per-image weights; sort highest-weight images first (Character1 = most prominent)
    try:
        weights = json.loads(image_weights)
    except Exception:
        weights = [1.0] * len(images)

    while len(weights) < len(images):
        weights.append(1.0)

    # Pair images with their weights and sort descending so Character1 = highest weight
    image_pairs = sorted(zip(weights, images), key=lambda x: x[0], reverse=True)
    sorted_weights, sorted_images = zip(*image_pairs)

    # Augment prompt with motion keywords based on motion_strength
    motion_tag = ""
    if motion_strength < 0.3:
        motion_tag = " Subtle, gentle motion."
    elif motion_strength < 0.6:
        motion_tag = " Smooth, natural motion."
    elif motion_strength < 0.85:
        motion_tag = " Dynamic, expressive motion."
    else:
        motion_tag = " Highly dynamic, energetic motion with strong movement."

    # Auto-inject Character references if not already in prompt
    char_refs = [f"Character{i+1}" for i in range(len(sorted_images))]
    full_prompt = prompt
    if not any(c in prompt for c in char_refs):
        ref_str = ", ".join(char_refs[:len(sorted_images)])
        full_prompt = f"Featuring {ref_str}. {prompt}"
    full_prompt += motion_tag

    async def event_stream():
        try:
            # Step 1: Upload images
            yield sse({"status": "uploading", "message": f"Uploading {len(sorted_images)} image(s) to fal storage…"})

            image_urls = []
            for i, img in enumerate(sorted_images):
                suffix = os.path.splitext(img.filename or ".jpg")[1] or ".jpg"
                content = await img.read()
                yield sse({"status": "uploading", "message": f"Uploading image {i+1}/{len(sorted_images)}…"})
                url = await upload_image(content, suffix)
                image_urls.append(url)

            yield sse({"status": "submitted", "message": "All images uploaded. Submitting to fal.ai…"})

            # Step 2: Submit job
            arguments = {
                "prompt": full_prompt,
                "image_urls": image_urls,
                "video_urls": [],
                "aspect_ratio": aspect_ratio,
                "resolution": resolution,
                "duration": duration,
                "negative_prompt": negative_prompt,
                "enable_safety_checker": False,
                "enable_prompt_expansion": False,
            }

            handle = await fal_client.submit_async(FAL_ENDPOINT, arguments=arguments)
            yield sse({"status": "queued", "message": "Job queued. Waiting for a worker…", "request_id": handle.request_id})

            # Step 3: Poll for progress
            logs_seen = 0
            async for event in handle.iter_events(with_logs=True):
                if isinstance(event, fal_client.Queued):
                    yield sse({
                        "status": "queued",
                        "message": f"Position in queue: {event.position}",
                        "position": event.position,
                    })
                elif isinstance(event, (fal_client.InProgress, fal_client.Completed)):
                    new_logs = event.logs[logs_seen:]
                    logs_seen = len(event.logs)
                    for log in new_logs:
                        yield sse({"status": "processing", "message": log.get("message", "")})

            # Step 4: Get result
            result = await handle.get()

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
            })

        except Exception as exc:
            yield sse({"status": "error", "message": str(exc)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
