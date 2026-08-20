import gc
import numpy as np
import cv2
import torch
from PIL import Image

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
from diffusers.models.attention_processor import Attention
from app.services.hand_detector import detect_hands

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
MASK_IMAGE = "sample-images/nail_mask_2.webp"
REFERENCE_IMAGE = "sample-images/sample.png"

base_image = load_image(ORIGINAL_IMAGE).convert("RGB")
with open(ORIGINAL_IMAGE, "rb") as f:
    image_bytes = f.read()

hands_data = detect_hands(image_bytes)
if not hands_data:
    print("No hands detected in the image. Please provide an image with visible hands.")
    exit(1)

mask_image = load_image(MASK_IMAGE).convert("L")


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
# 4. Load SDXL Inpainting + ControlNet
# -------------------------------------------------------
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=dtype
).to(device)

pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    controlnet=controlnet,
    torch_dtype=dtype
).to(device)

# CPU memory optimizations
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

generator = torch.Generator(device="cpu").manual_seed(42)

# Cap processing size to avoid OOM on Intel Mac CPU.
MAX_SIDE = 512
orig_w, orig_h = base_image.size
scale = min(1.0, MAX_SIDE / min(orig_w, orig_h))
proc_w, proc_h = (int(orig_w * scale), int(orig_h * scale))
proc_w -= proc_w % 64
proc_h -= proc_h % 64

base_proc = base_image.resize((proc_w, proc_h), Image.LANCZOS)
mask_proc = mask_processed.resize((proc_w, proc_h), Image.NEAREST)

# Soften mask boundary to avoid dark edge artifacts from hard inpainting mask
mask_proc_np = np.array(mask_proc).astype(np.float32) / 255.0
mask_proc_np = cv2.GaussianBlur(mask_proc_np, (21, 21), 5)
mask_proc = Image.fromarray((mask_proc_np * 255).astype(np.uint8))

# Canny control image at processing resolution
base_cv = cv2.cvtColor(np.array(base_proc), cv2.COLOR_RGB2BGR)
canny = cv2.Canny(base_cv, 50, 150)
canny_rgb = np.stack([canny] * 3, axis=-1)
control_image = Image.fromarray(canny_rgb)

# -------------------------------------------------------
# 5. Prompt
# -------------------------------------------------------
prompt = (
    "Professional manicure with glossy polished fingernails. "
    "Photorealistic, highly detailed, natural lighting, same angle as original."
)

negative_prompt = (
    "bare nails, damaged nails, blurry, low quality"
)

# -------------------------------------------------------
# 6. Inpaint with ControlNet to get nail shape + lighting
# -------------------------------------------------------
result_proc = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=base_proc,
    mask_image=mask_proc,
    control_image=control_image,
    num_inference_steps=2,
    guidance_scale=0.5,
    strength=0.95,
    generator=generator,
    controlnet_conditioning_scale=0.5,
).images[0]

# Resize back to original size
if result_proc.size != base_image.size:
    result = result_proc.resize(base_image.size, Image.LANCZOS)
else:
    result = result_proc

# -------------------------------------------------------
# 7. Blend actual pattern from sample.png onto result
#     - Perspective warp to match nail shape (handles flipped/narrow/wide nails)
#     - No tiling: sample.png is warped to fit each nail's rotated rectangle
#     - More opaque polish look with soft feathering at skin edges
# -------------------------------------------------------
ref = Image.open(REFERENCE_IMAGE).convert("RGBA")

# Use the original dilated mask (not blurred) for pattern placement
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
final_np = np.array(result).copy()

# Compute shading map once from generated nails (preserves 3D curvature)
base_gray = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2GRAY) / 255.0

# Build fingertip lookup from detected hands: (x_pixel, y_pixel) -> angle_deg
fingertip_lookup = {}
if hands_data:
    for hand in hands_data:
        for ft in hand.get("fingertips", []):
            fx = ft["x"] * orig_w
            fy = ft["y"] * orig_h
            fingertip_lookup[(fx, fy)] = ft.get("a", -90)

# Collect valid nail components and do one-to-one matching to fingertips
valid_components = []
for comp_i in range(1, num_labels):
    cx, cy, cw, ch, carea = stats[comp_i]
    if carea < 50 or cw <= 0 or ch <= 0:
        continue
    valid_components.append((comp_i, cx + cw / 2, cy + ch / 2))

assigned_angles = {}  # component_index -> angle
used_fingertips = set()

if fingertip_lookup:
    pairs = []
    for comp_idx, comp_cx, comp_cy in valid_components:
        for ft_pos, ft_angle in fingertip_lookup.items():
            dist = (ft_pos[0] - comp_cx) ** 2 + (ft_pos[1] - comp_cy) ** 2
            pairs.append((dist, comp_idx, ft_pos, ft_angle))

    pairs.sort(key=lambda p: p[0])

    for dist, comp_idx, ft_pos, ft_angle in pairs:
        if comp_idx not in assigned_angles and ft_pos not in used_fingertips:
            assigned_angles[comp_idx] = ft_angle
            used_fingertips.add(ft_pos)

for i in range(1, num_labels):
    x, y, w, h, area = stats[i]
    if area < 50 or w <= 0 or h <= 0:
        continue

    # Full-resolution component mask for this nail
    comp_mask_full = (labels == i).astype(np.uint8) * 255

    # Find contours to get nail geometry
    contours, _ = cv2.findContours(comp_mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        continue
    contour = max(contours, key=cv2.contourArea)

    # Get rotated bounding rectangle for this nail
    # This handles flipped, narrow, or wide nails via perspective
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.float32)

    # Order points: top-left, top-right, bottom-right, bottom-left
    sum_pts = box.sum(axis=1)
    diff_pts = box[:, 0] - box[:, 1]
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = box[np.argmin(sum_pts)]
    ordered[2] = box[np.argmax(sum_pts)]
    ordered[1] = box[np.argmin(diff_pts)]
    ordered[3] = box[np.argmax(diff_pts)]

    # Use pre-assigned angle for this component, or fall back to default
    nail_angle = -assigned_angles.get(i, -90)
    print(f"Nail {i}: angle {nail_angle:.2f} degrees (fingertip: {assigned_angles.get(i, -90):.2f})")

    # Rotate sample.png to match nail angle
    rotated = ref.rotate(nail_angle, resample=Image.BICUBIC, expand=True)
    rot_w, rot_h = rotated.size
    ref_np_rot = np.array(rotated)

    # Perspective transform: map sample.png to the nail's rotated rectangle
    src_pts = np.array([
        [0, 0],
        [rot_w, 0],
        [rot_w, rot_h],
        [0, rot_h]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, ordered)
    warped = cv2.warpPerspective(ref_np_rot, M, (orig_w, orig_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)

    # Extract RGB and alpha from warped pattern
    if warped.shape[2] == 4:
        tile_rgb = warped[:, :, :3]
        alpha_warped = warped[:, :, 3] / 255.0
    else:
        tile_rgb = warped
        alpha_warped = np.ones((orig_h, orig_w), dtype=np.float32)

    # Feather the mask for soft edge blending (skin boundary)
    # Wider blur = nail gradually mixes into skin instead of sitting on top
    mask_blur = cv2.GaussianBlur(comp_mask_full, (51, 51), 21)
    mask_alpha = mask_blur / 255.0

    # Combined alpha
    # Use sample.png alpha only to exclude transparent padding,
    # but nail interior is fully opaque (real polish) with wide soft falloff into skin.
    valid_pattern = alpha_warped > 0.05
    blend_alpha = np.clip(mask_alpha * valid_pattern.astype(np.float32) * 1.2, 0, 1)

    # Blend: pattern is mostly opaque (realistic polish),
    # but fades at skin edges via feathered mask
    for c in range(3):
        pattern_ch = tile_rgb[:, :, c].astype(np.float32) / 255.0
        base_ch = final_np[:, :, c].astype(np.float32) / 255.0
        
        # Multiply blend: pattern color * base lighting/shading
        # This preserves 3D curvature and highlights from the generated nail
        blended = pattern_ch * (base_gray * 0.5 + 0.5)
        
        # Soft composite: pattern is mostly opaque (realistic polish),
        # but fades at skin edges via feathered mask
        final_np[:, :, c] = (blended * blend_alpha + base_ch * (1 - blend_alpha)) * 255

final = Image.fromarray(final_np.astype(np.uint8))


# -------------------------------------------------------
# 8. Save
# -------------------------------------------------------
OUTPUT = "sample-images/output_painted_nails.png"
final.save(OUTPUT)

print(f"Saved: {OUTPUT}")

gc.collect()

final.show()