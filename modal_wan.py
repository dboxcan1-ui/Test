"""Modal deployment for Phantom-Wan reference-to-video + persistent web app.

Setup:
    pip install modal
    modal token new
    modal run modal_wan.py::download_model   # download weights once
    modal deploy modal_wan.py                # deploy app

Volumes created automatically:
    wan-models   – Phantom + Wan2.1 base weights
    ref2vid-data – videos, refs, prompts, logs (persists across deploys)
"""
import os
import modal
from pathlib import Path

app = modal.App("ref2vid-wan")

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR = Path("/models")
DATA_DIR  = Path("/data")

# Phantom needs Wan2.1-T2V-1.3B (VAE + text encoder) + its own weights
WAN_BASE_ID     = "Wan-AI/Wan2.1-T2V-1.3B"
PHANTOM_ID      = "bytedance-research/Phantom"
WAN_BASE_SUBDIR = "wan-base"
PHANTOM_SUBDIR  = "phantom"

# ── Volumes ───────────────────────────────────────────────────────────────────
model_volume = modal.Volume.from_name("wan-models",   create_if_missing=True)
data_volume  = modal.Volume.from_name("ref2vid-data", create_if_missing=True)

# ── GPU image ─────────────────────────────────────────────────────────────────
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .run_commands("git clone https://github.com/Phantom-video/Phantom /opt/phantom")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "huggingface_hub>=0.25.0",
        "Pillow>=10.0.0",
        "safetensors",
        "sentencepiece",
        "imageio[ffmpeg]>=2.34.0",
        "easydict",
        "ftfy",
        "numpy",
    )
)


# ── Aspect ratio → Phantom --size (WxH) ──────────────────────────────────────
ASPECT_SIZES = {
    "9:16": "480*832",
    "16:9": "832*480",
    "1:1":  "480*480",
    "4:3":  "640*480",
    "3:4":  "480*640",
}

# ── Secrets ───────────────────────────────────────────────────────────────────
_dotenv_secrets = [modal.Secret.from_dotenv()] if Path(".env").exists() else []


# ── Model download (run once: modal run modal_wan.py::download_model) ─────────
@app.function(
    image=gpu_image,
    volumes={str(MODEL_DIR): model_volume},
    secrets=_dotenv_secrets,
    timeout=3600,
)
def download_model():
    """Download Phantom-Wan weights into the persistent volume.

    Run once before first inference:
        modal run modal_wan.py::download_model

    Downloads two repos:
      1. Wan-AI/Wan2.1-T2V-1.3B  – VAE + text encoder used by Phantom
      2. bytedance-research/Phantom – Phantom-Wan-14B weights
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    model_volume.reload()

    repos = [
        (WAN_BASE_ID, WAN_BASE_SUBDIR, "diffusion_pytorch_model.safetensors"),
        (PHANTOM_ID,  PHANTOM_SUBDIR,  "Phantom-Wan-14B.pth"),
    ]

    for repo_id, subdir, sentinel in repos:
        model_path = MODEL_DIR / subdir
        if (model_path / sentinel).exists():
            print(f"{repo_id} already present at {model_path}, skipping.")
            continue

        all_files = sorted(list_repo_files(repo_id))
        print(f"Downloading {repo_id} ({len(all_files)} files) → {model_path}")

        for i, filename in enumerate(all_files, 1):
            dest = model_path / filename
            if dest.exists() and dest.stat().st_size > 0:
                print(f"  [{i}/{len(all_files)}] Already have {filename}, skipping")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"  [{i}/{len(all_files)}] {filename} …")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(model_path),
                local_dir_use_symlinks=False,
            )
            model_volume.commit()

    print("Download complete.")


# ── PhantomGenerator ──────────────────────────────────────────────────────────
@app.cls(
    gpu="A100-80GB",
    image=gpu_image,
    volumes={
        str(MODEL_DIR): model_volume,
        str(DATA_DIR):  data_volume,
    },
    timeout=600,
    startup_timeout=600,
    scaledown_window=600,
)
class WanGenerator:
    @modal.enter()
    def load(self):
        import sys
        sys.path.insert(0, "/opt/phantom")
        model_volume.reload()

        wan_path     = MODEL_DIR / WAN_BASE_SUBDIR
        phantom_path = MODEL_DIR / PHANTOM_SUBDIR

        if not (phantom_path / "Phantom-Wan-14B.pth").exists():
            raise RuntimeError(
                "Phantom weights not found. "
                "Run `modal run modal_wan.py::download_model` first."
            )

        self.wan_path     = str(wan_path)
        self.phantom_path = str(phantom_path)

    @modal.method()
    def generate(
        self,
        image_bytes: bytes,
        prompt: str,
        negative_prompt: str,
        aspect_ratio: str,
        duration: int,
    ) -> str:
        """Run Phantom-Wan inference and save the MP4 to the data volume.
        Returns the video_id (hex string) so the web endpoint can serve it."""
        import subprocess
        import tempfile
        import uuid

        size      = ASPECT_SIZES.get(aspect_ratio, "480*832")
        frame_num = duration * 24 + 1   # Phantom trained at 24 fps
        video_id  = uuid.uuid4().hex

        videos_dir = DATA_DIR / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        output_path = videos_dir / f"{video_id}.mp4"

        with tempfile.TemporaryDirectory() as tmp:
            ref_path = Path(tmp) / "ref.jpg"
            ref_path.write_bytes(image_bytes)

            cmd = [
                "python", "/opt/phantom/generate.py",
                "--task",         "s2v-14B",
                "--size",         size,
                "--frame_num",    str(frame_num),
                "--sample_fps",   "24",
                "--ckpt_dir",     self.wan_path,
                "--phantom_ckpt", self.phantom_path,
                "--ref_image",    str(ref_path),
                "--prompt",       prompt,
                "--save_file",    str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Phantom inference failed:\n{result.stderr[-3000:]}"
                )

        if not output_path.exists():
            raise RuntimeError("Phantom ran successfully but output file not found.")

        data_volume.commit()
        return video_id


