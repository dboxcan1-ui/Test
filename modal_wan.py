"""Modal deployment for WAN I2V video generation.

Setup:
    pip install modal
    modal token new              # one-time auth
    modal deploy modal_wan.py   # deploy the app

Environment variables needed on the FastAPI host:
    MODAL_TOKEN_ID
    MODAL_TOKEN_SECRET
"""
import modal
from pathlib import Path

app = modal.App("ref2vid-wan")

MODEL_DIR = Path("/models")
MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P"
MODEL_SUBDIR = "wan-i2v"

volume = modal.Volume.from_name("wan-models", create_if_missing=True)

image = (
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

# (height, width) for 480p model at each aspect ratio
ASPECT_DIMS = {
    "9:16": (832, 480),
    "16:9": (480, 832),
    "1:1":  (480, 480),
    "4:3":  (624, 832),
    "3:4":  (832, 624),
}


@app.cls(
    gpu="A100-40GB",
    image=image,
    volumes={str(MODEL_DIR): volume},
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
            volume.commit()

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
    ) -> bytes:
        import io
        import os
        import tempfile

        from diffusers.utils import export_to_video
        from PIL import Image

        height, width = ASPECT_DIMS.get(aspect_ratio, (832, 480))
        num_frames = duration * 16 + 1  # odd count at ~16 fps

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        output = self.pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=5.0,
            num_inference_steps=30,
        )

        frames = output.frames[0]
        tmp = tempfile.mktemp(suffix=".mp4")
        try:
            export_to_video(frames, tmp, fps=16)
            with open(tmp, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
