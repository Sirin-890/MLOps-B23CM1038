"""
2-Page Streamlit App for UNet Segmentation
Matches training code that uses:
  - segmentation_models_pytorch with resnet34 encoder
  - checkpoints/unet_model.pth
  - loss_curve.png / metrics_curve.png
  - CameraRGB / CameraMask directories
  - 23 classes
"""

import os
import numpy as np
import torch
from PIL import Image
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UNet Segmentation",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded",
)

# ── Constants (must match training code exactly) ──────────────────────────────
NUM_CLASSES = 23
IMG_SIZE    = (256, 256)
MODEL_PATH  = "checkpoints/unet_model.pth"
LOSS_PLOT   = "loss_curve.png"
METRIC_PLOT = "metrics_curve.png"

# CARLA-style 23-class palette
PALETTE = np.array([
    [0,   0,   0  ], [70,  70,  70 ], [190,153, 153],
    [250,170, 160 ], [200,130,   0 ], [220, 20,  60 ],
    [255,  0,   0 ], [  0,  0, 142 ], [  0,  0,  90 ],
    [  0, 60, 100 ], [  0,  0, 230 ], [119, 11,  32 ],
    [  0, 80, 100 ], [  0,  0, 192 ], [128, 64, 255 ],
    [  0,182,   0 ], [220,220,   0 ], [107,142,  35 ],
    [152,251, 152 ], [ 70,130, 180 ], [220,220, 220 ],
    [244, 35, 232 ], [255,255, 255 ],
], dtype=np.uint8)

CLASS_NAMES = [
    "Unlabeled","Building","Fence","Other","Pedestrian",
    "Pole","RoadLine","Road","SideWalk","Vegetation",
    "Vehicles","Wall","TrafficSign","Sky","Ground",
    "Bridge","RailTrack","GuardRail","TrafficLight","Static",
    "Dynamic","Water","Terrain",
]

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px;
}
.caption-tag {
    background: #21262d; border: 1px solid #30363d; border-radius: 6px;
    padding: 4px 10px; font-size: 12px; font-family: 'Space Mono', monospace;
    color: #79c0ff; display: inline-block; margin-bottom: 6px;
}
.section-header {
    font-family: 'Space Mono', monospace; font-size: 13px;
    letter-spacing: 2px; text-transform: uppercase; color: #58a6ff;
    padding: 8px 0 4px; border-bottom: 1px solid #30363d; margin-bottom: 16px;
}
.info-box {
    background: #161b22; border-left: 3px solid #58a6ff;
    border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 8px 0; font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


# ── Model loader ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    try:
        import segmentation_models_pytorch as smp
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES,
        )
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            model.eval()
            return model, True
        return model, False
    except ImportError:
        st.error("`segmentation_models_pytorch` not installed. Run: `pip install segmentation-models-pytorch`")
        return None, False


# ── Helpers ────────────────────────────────────────────────────────────────────
def preprocess(img: Image.Image) -> torch.Tensor:
    """Matches training: float32, /255, permute → (1,3,H,W)"""
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.tensor(arr).permute(2, 0, 1).unsqueeze(0)


def mask_to_color(mask_np: np.ndarray) -> np.ndarray:
    h, w = mask_np.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(NUM_CLASSES):
        color[mask_np == cls] = PALETTE[cls]
    return color


@torch.no_grad()
def predict(model, img: Image.Image) -> np.ndarray:
    tensor = preprocess(img)
    out = model(tensor)
    return out.argmax(dim=1).squeeze(0).cpu().numpy()


def compute_metrics(preds_np, targets_np):
    """Identical logic to training code's compute_metrics()."""
    preds   = torch.tensor(preds_np.astype(np.int64)).view(-1)
    targets = torch.tensor(targets_np.astype(np.int64)).view(-1)
    ious, dices = [], []
    for cls in range(NUM_CLASSES):
        pred_inds   = (preds == cls)
        target_inds = (targets == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union        = (pred_inds | target_inds).sum().item()
        if union == 0:
            continue
        iou  = intersection / union
        dice = (2 * intersection) / (pred_inds.sum().item() + target_inds.sum().item() + 1e-6)
        ious.append(iou); dices.append(dice)
    return (
        round(sum(ious)  / len(ious)  if ious  else 0.0, 4),
        round(sum(dices) / len(dices) if dices else 0.0, 4),
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 UNet Seg")
    st.markdown("---")
    page = st.radio("Navigate", ["📊  Training Results", "🖼️  Run Inference"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Model:** UNet + ResNet34")
    st.markdown("**Classes:** 23 (CARLA)")
    st.markdown("**Epochs:** 20  |  **LR:** 1e-3")
    st.markdown("**Loss:** CrossEntropyLoss")
    st.markdown("**Optim:** Adam")
    st.markdown("---")
    if os.path.exists(MODEL_PATH):
        st.success("✅ Weights loaded")
    else:
        st.warning(f"⚠️ No weights at\n`{MODEL_PATH}`")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Training Results
# ══════════════════════════════════════════════════════════════════════════════
if "Training" in page:
    st.markdown("# 📊 Training Results")
    st.markdown('<p class="section-header">Test Set Metrics</p>', unsafe_allow_html=True)

    # Try loading saved test metrics
    test_miou, test_mdice = None, None
    for path in ["Question2/metrics.json", "metrics.json"]:
        if os.path.exists(path):
            import json
            with open(path) as f:
                saved = json.load(f)
            test_miou  = saved.get("test_miou",  saved.get("val_iou"))
            test_mdice = saved.get("test_mdice", saved.get("val_dice"))
            break

    col1, col2, col3 = st.columns(3)
    col1.metric("Architecture", "UNet + ResNet34")
    col2.metric("Test mIoU",  f"{test_miou:.4f}"  if test_miou  is not None else "Run train.py")
    col3.metric("Test mDice", f"{test_mdice:.4f}" if test_mdice is not None else "Run train.py")

    if test_miou is None:
        st.markdown(
            '<div class="info-box">💡 Add <code>test_miou</code> & <code>test_mdice</code> to '
            '<code>metrics.json</code> after running <code>train.py</code>.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<p class="section-header">Training Curves</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<span class="caption-tag">loss_curve.png</span>', unsafe_allow_html=True)
        if os.path.exists(LOSS_PLOT):
            st.image(LOSS_PLOT, use_column_width=True)
        else:
            st.info(f"`{LOSS_PLOT}` not found — run `train.py` first.")

    with c2:
        st.markdown('<span class="caption-tag">metrics_curve.png</span>', unsafe_allow_html=True)
        if os.path.exists(METRIC_PLOT):
            st.image(METRIC_PLOT, use_column_width=True)
        else:
            st.info(f"`{METRIC_PLOT}` not found — run `train.py` first.")

    st.markdown('<p class="section-header">Class Colour Legend</p>', unsafe_allow_html=True)
    cols = st.columns(6)
    for i, name in enumerate(CLASS_NAMES):
        c = PALETTE[i]
        hex_c = f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"
        txt = "#000" if c.mean() > 128 else "#fff"
        cols[i % 6].markdown(
            f'<span style="background:{hex_c};color:{txt};padding:2px 8px;'
            f'border-radius:4px;font-size:11px;">{i}: {name}</span>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Inference
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("# 🖼️ Run Inference")
    st.markdown(
        '<div class="info-box">Upload <b>up to 4 RGB images</b> from the test set. '
        'Optionally upload ground-truth masks (same order) for side-by-side comparison '
        'and per-image mIoU / mDice.</div>',
        unsafe_allow_html=True,
    )

    model, model_loaded = load_model()
    if not model_loaded:
        st.error(f"Weights not found at `{MODEL_PATH}`. Run `train.py` to generate them.")
        st.stop()

    st.markdown("---")
    c_rgb, c_mask = st.columns(2)
    with c_rgb:
        rgb_uploads = st.file_uploader(
            "📂 Upload RGB Images (max 4)",
            type=["png","jpg","jpeg"],
            accept_multiple_files=True,
            key="rgb_up",
        )
    with c_mask:
        mask_uploads = st.file_uploader(
            "🗂️ Upload Ground-Truth Masks (optional)",
            type=["png","jpg","jpeg"],
            accept_multiple_files=True,
            key="mask_up",
        )

    if not rgb_uploads:
        st.markdown("---")
        st.info("👆 Upload RGB images above to see predictions.")
        st.stop()

    n = min(len(rgb_uploads), 4)
    if len(rgb_uploads) > 4:
        st.warning("Only the first 4 images will be processed.")

    st.markdown(f"---\n### Results — {n} image(s)")

    for i in range(n):
        img        = Image.open(rgb_uploads[i]).convert("RGB")
        pred_mask  = predict(model, img)
        pred_color = mask_to_color(pred_mask)
        has_gt     = mask_uploads and i < len(mask_uploads)

        # ── Ground-truth processing (mirrors training code) ──
        if has_gt:
            gt_raw = Image.open(mask_uploads[i])
            gt_np  = np.array(gt_raw)
            if len(gt_np.shape) == 3:          # same fix as training code
                gt_np = gt_np[:, :, 0]
            gt_np = np.clip(gt_np, 0, NUM_CLASSES - 1)
            gt_np_resized = np.array(
                Image.fromarray(gt_np.astype(np.uint8)).resize(IMG_SIZE, Image.NEAREST)
            )
            gt_color       = mask_to_color(gt_np_resized)
            miou, mdice    = compute_metrics(pred_mask, gt_np_resized)

        st.markdown(
            f'<span class="caption-tag">Image {i+1} — {rgb_uploads[i].name}</span>',
            unsafe_allow_html=True,
        )

        if has_gt:
            col_a, col_b, col_c = st.columns(3)
            col_a.image(img.resize(IMG_SIZE), caption="RGB Input",         use_column_width=True)
            col_b.image(gt_color,              caption="Ground-Truth Mask", use_column_width=True)
            col_c.image(pred_color,            caption="Predicted Mask",    use_column_width=True)
            m1, m2, _ = st.columns([1, 1, 2])
            m1.metric("mIoU",  f"{miou:.4f}")
            m2.metric("mDice", f"{mdice:.4f}")
        else:
            col_a, col_b = st.columns(2)
            col_a.image(img.resize(IMG_SIZE), caption="RGB Input",      use_column_width=True)
            col_b.image(pred_color,            caption="Predicted Mask", use_column_width=True)

        # Detected class tags
        unique_cls = np.unique(pred_mask)
        tags = "".join(
            f'<span style="background:{("#%02x%02x%02x" % tuple(PALETTE[c].tolist()))};'
            f'color:{"#000" if PALETTE[c].mean()>128 else "#fff"};'
            f'padding:2px 7px;border-radius:4px;margin:2px;font-size:11px;display:inline-block;">'
            f'{CLASS_NAMES[c]}</span>'
            for c in unique_cls if c < NUM_CLASSES
        )
        st.markdown(f"**Detected:** {tags}", unsafe_allow_html=True)
        st.markdown("---")