import gc
import numpy as np
import cv2
import torch
from PIL import Image, ImageFilter

# Fix for macOS CPU where torch.xpu is unavailable but diffusers checks for it.
if not hasattr(torch, "xpu"):
    class _DummyXPU:
        @staticmethod
        def empty_cache():
            pass
        @staticmethod
        def device_count():
            return 0
        @staticmethod
        def manual_seed(*args, **kwargs):
            pass
        @staticmethod
        def is_available():
            return False
    torch.xpu = _DummyXPU()

from diffusers import AutoPipelineForInpainting
from diffusers.models.attention_processor import Attention

# Patch Attention.forward to unwrap encoder_hidden_states tuples
# returned by IP-Adapter image encoders on older diffusers versions.
_orig_forward = Attention.forward

def _patched_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
    if isinstance(encoder_hidden_states, tuple):
        encoder_hidden_states = encoder_hidden_states[0]
    return _orig_forward(self, hidden_states, encoder_hidden_states, attention_mask, **cross_attention_kwargs)

Attention.forward = _patched_forward
from diffusers.utils import load_image

# -------------------------------------------------------
# 1. CPU Setup (Intel Mac)
# -------------------------------------------------------
device = "cpu"
dtype = torch.float32

# -------------------------------------------------------
# 2. Input Images
# -------------------------------------------------------
ORIGINAL_IMAGE = "sample-images/hand.webp"
MASK_IMAGE = "sample-images/nail_mask.webp"
REFERENCE_IMAGE = "sample-images/sample.png"

base_image = load_image(ORIGINAL_IMAGE).convert("RGB")
mask_image = load_image(MASK_IMAGE).convert("L")
reference_image = load_image(REFERENCE_IMAGE).convert("RGB")


# Resize mask to match original
if mask_image.size != base_image.size:
    mask_image = mask_image.resize(base_image.size, Image.NEAREST)

# -------------------------------------------------------
# 3. Clean & Expand Mask
# -------------------------------------------------------
mask_np = np.array(mask_image)

# Binary mask
mask_np = np.where(mask_np > 30, 255, 0).astype(np.uint8)

# Slight dilation so polish reaches edges
kernel = np.ones((5, 5), np.uint8)
mask_np = cv2.dilate(mask_np, kernel, iterations=2)

if np.count_nonzero(mask_np) == 0:
    raise ValueError("Mask contains no white pixels.")

mask_processed = Image.fromarray(mask_np)

# -------------------------------------------------------
# 4. Load SDXL Inpainting + IP Adapter
# -------------------------------------------------------
pipe = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=dtype
).to(device)

# Load IP-Adapter (Reference Image Guidance)
pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter_sdxl.bin",
)

# How strongly to follow the sample image
pipe.set_ip_adapter_scale(0.9)

# Optional memory optimization for CPU
pipe.enable_attention_slicing()

generator = torch.Generator(device="cpu").manual_seed(42)

# -------------------------------------------------------
# 5. Prompt
# -------------------------------------------------------
prompt = (
    "Professional manicure with glossy polished fingernails. "
    "Match the color, texture, finish and design of the reference nails. "
    "Photorealistic, highly detailed."
)

negative_prompt = (
    "natural nails, bare nails, damaged nails, blurry, low quality, extra fingers"
)

# -------------------------------------------------------
# 6. Inpaint
# -------------------------------------------------------
result = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=base_image,
    mask_image=mask_processed,
    ip_adapter_image=reference_image,
    num_inference_steps=30,
    guidance_scale=8.5,
    strength=1.0,
    generator=generator,
).images[0]

# -------------------------------------------------------
# 7. Feather Blend
# -------------------------------------------------------
mask_feather = (
    mask_processed
    .filter(ImageFilter.GaussianBlur(radius=3))
    .convert("L")
)

if result.size != base_image.size:
    result = result.resize(base_image.size, Image.LANCZOS)

final = Image.composite(result, base_image, mask_feather)

# -------------------------------------------------------
# 8. Save
# -------------------------------------------------------
OUTPUT = "sample-images/output_painted_nails.png"
final.save(OUTPUT)

print(f"Saved: {OUTPUT}")

gc.collect()

final.show()