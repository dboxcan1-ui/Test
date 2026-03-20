"""Modal deployment for WAN I2V video generation + persistent web app.

Setup:
    pip install modal
    modal token new              # one-time auth
    modal deploy modal_wan.py    # deploy everything

Volumes created automatically:
    wan-models   – model weights cache
    ref2vid-data – videos, refs, prompts, logs (persists across deploys)
"""
import os
import modal
from pathlib import Path

app = modal.App("ref2vid-wan")

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR = Path("/models")
DATA_DIR  = Path("/data")

MODEL_ID     = "Wan-AI/Wan2.1-I2V-14B-480P"
MODEL_SUBDIR = "wan-i2v"

# ── Volumes ───────────────────────────────────────────────────────────────────
model_volume = modal.Volume.from_name("wan-models",   create_if_missing=True)
data_volume  = modal.Volume.from_name("ref2vid-data", create_if_missing=True)

# ── GPU image ─────────────────────────────────────────────────────────────────
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "diffusers>=0.31.0",
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "huggingface_hub>=0.25.0",
        "imageio[ffmpeg]>=2.34.0",
        "Pillow>=10.0.0",
        "safetensors",
        "sentencepiece",
    )
)

# ── Web image ─────────────────────────────────────────────────────────────────
web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi>=0.115.0",
        "uvicorn[standard]",
        "python-multipart",
        "Pillow>=10.0.0",
        "fal-client>=0.5.0",
        "modal>=0.73.0",
        "python-dotenv",
    )
    .add_local_file("main.py",    "/app/main.py")
    .add_local_file("index.html", "/app/index.html")
)

# ── Aspect ratio dims (height, width) for 480p model ─────────────────────────
ASPECT_DIMS = {
    "9:16": (832, 480),
    "16:9": (480, 832),
    "1:1":  (480, 480),
    "4:3":  (624, 832),
    "3:4":  (832, 624),
}


# ── WanGenerator ─────────────────────────────────────────────────────────────
@app.cls(
    gpu="A100-40GB",
    image=gpu_image,
    volumes={
        str(MODEL_DIR): model_volume,
        str(DATA_DIR):  data_volume,
    },
    timeout=600,
    scaledown_window=120,
)
class WanGenerator:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import WanImageToVideoPipeline
        from huggingface_hub import snapshot_download

        model_path = MODEL_DIR / MODEL_SUBDIR
        if not (model_path / "model_index.json").exists():
            snapshot_download(repo_id=MODEL_ID, local_dir=str(model_path))
            model_volume.commit()

        self.pipe = WanImageToVideoPipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
        ).to("cuda")

    @modal.method()
    def generate(
        self,
        image_bytes: bytes,
        prompt: str,
        negative_prompt: str,
        aspect_ratio: str,
        duration: int,
    ) -> str:
        """Run inference and save the MP4 to the data volume.
        Returns the video_id (hex string) so the web endpoint can serve it."""
        import io
        import tempfile
        import uuid
        from diffusers.utils import export_to_video
        from PIL import Image

        height, width = ASPECT_DIMS.get(aspect_ratio, (832, 480))
        num_frames = duration * 16 + 1

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        out = self.pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=5.0,
            num_inference_steps=30,
        )

        # ── Save video to shared data volume ──────────────────────────────────
        frames    = out.frames[0]
        video_id  = uuid.uuid4().hex
        videos_dir = DATA_DIR / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)

        tmp = tempfile.mktemp(suffix=".mp4")
        try:
            export_to_video(frames, tmp, fps=16)
            (videos_dir / f"{video_id}.mp4").write_bytes(Path(tmp).read_bytes())
        finally:
            if Path(tmp).exists():
                Path(tmp).unlink()

        data_volume.commit()
        return video_id


# ── Web endpoint ──────────────────────────────────────────────────────────────
_web_secrets = [modal.Secret.from_dotenv()] if Path(".env").exists() else []


@app.function(
    image=web_image,
    volumes={str(DATA_DIR): data_volume},
    secrets=_web_secrets,
    timeout=900,
    scaledown_window=60,
    allow_concurrent_inputs=20,
)
@modal.asgi_app()
def web():
    import os
    import sys

    sys.path.insert(0, "/app")

    # Set DATA_DIR before importing main so module-level path init picks it up
    os.environ["DATA_DIR"] = str(DATA_DIR)

    import main as _main

    # Inject volume reference so main.py can reload/commit
    _main._volume = data_volume

    # Override paths in case the module was already cached
    _main.DATA_DIR    = DATA_DIR
    _main.REFS_DIR    = DATA_DIR / "refs"
    _main.VIDEOS_DIR  = DATA_DIR / "videos"
    _main.PROMPTS_FILE = DATA_DIR / "prompts.json"
    _main.REFS_DIR.mkdir(parents=True, exist_ok=True)
    _main.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    return _main.app
