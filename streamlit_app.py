import os

# ✅ Silence TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import streamlit as st
import numpy as np
import tensorflow as tf
import json
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image

# =========================
# 🔥 Load Model (cache)
# =========================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model_finetuned.keras")

    dummy_input = tf.zeros((1, 224, 224, 3))
    model(dummy_input)

    base_model = None
    for layer in model.layers:
        if "efficientnet" in layer.name.lower():
            base_model = layer
            break

    base_model(dummy_input)

    # Find last conv layer
    last_conv_layer_name = None
    for layer in reversed(base_model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            last_conv_layer_name = layer.name
            break

    # Find base_model index
    base_model_index = None
    for i, layer in enumerate(model.layers):
        if layer.name == base_model.name:
            base_model_index = i
            break

    return model, base_model, last_conv_layer_name, base_model_index


model, base_model, last_conv_layer_name, base_model_index = load_model()

# =========================
# 📂 Load class names
# =========================
with open("class_names.json", "r") as f:
    CLASS_NAMES = json.load(f)

# =========================
# 🔥 Grad-CAM
# =========================
def make_gradcam_heatmap(img_array):
    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    conv_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=last_conv_layer.output
    )

    x = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    y = x

    for layer in model.layers[base_model_index + 1:]:
        y = layer(y)

    classifier_model = tf.keras.models.Model(x, y)

    with tf.GradientTape() as tape:
        conv_outputs = conv_model(img_array)
        tape.watch(conv_outputs)

        predictions = classifier_model(conv_outputs)

        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def generate_gradcam(image_array, original_image):
    heatmap = make_gradcam_heatmap(image_array)

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    heatmap_blurred = cv2.GaussianBlur(heatmap, (11, 11), 0)

    img = original_image.resize((224, 224))
    img = np.array(img)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    heatmap_uint8 = np.uint8(255 * heatmap_blurred)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    alpha = heatmap_blurred[..., np.newaxis]
    alpha = np.power(alpha, 0.6)

    superimposed = img_bgr * (1 - alpha * 0.75) + colored * (alpha * 0.75)
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)

    return cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)


# =========================
# 🧪 Preprocess
# =========================
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    return img_array


# =========================
# 🎨 UI
# =========================
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Dark clinical theme */
  [data-testid="stAppViewContainer"] {
      background: #0d1117;
      color: #e6edf3;
  }
  [data-testid="stHeader"] { background: transparent; }

  /* Section card */
  .card {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 12px;
      padding: 1.4rem 1.6rem;
      margin-bottom: 1.2rem;
  }

  /* Side-by-side scan panels */
  .scan-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
  }
  .scan-panel {
      background: #0d1117;
      border: 1px solid #30363d;
      border-radius: 10px;
      overflow: hidden;
  }
  .scan-panel img {
      width: 100%;
      display: block;
  }
  .scan-label {
      font-size: 0.72rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: #8b949e;
      text-align: center;
      padding: 6px 0 8px;
      font-family: monospace;
  }

  /* Prediction badge */
  .verdict-tumor {
      display: inline-flex; align-items: center; gap: 8px;
      background: #3d1f1f; border: 1px solid #f85149;
      color: #f85149; border-radius: 8px;
      padding: 0.5rem 1.1rem; font-size: 1.05rem; font-weight: 700;
      font-family: monospace; letter-spacing: 0.05em;
  }
  .verdict-notumor {
      display: inline-flex; align-items: center; gap: 8px;
      background: #1a2e1a; border: 1px solid #3fb950;
      color: #3fb950; border-radius: 8px;
      padding: 0.5rem 1.1rem; font-size: 1.05rem; font-weight: 700;
      font-family: monospace; letter-spacing: 0.05em;
  }
  .conf-text {
      font-family: monospace; font-size: 0.85rem;
      color: #8b949e; margin-top: 6px;
  }

  /* Progress bars */
  .prob-row { margin-bottom: 10px; }
  .prob-label {
      display: flex; justify-content: space-between;
      font-family: monospace; font-size: 0.78rem;
      color: #8b949e; margin-bottom: 4px;
  }
  .prob-bar-bg {
      background: #21262d; border-radius: 4px; height: 8px; overflow: hidden;
  }
  .prob-bar-fill {
      height: 100%; border-radius: 4px;
      background: linear-gradient(90deg, #1f6feb, #58a6ff);
      transition: width 0.6s ease;
  }
  .prob-bar-fill.high { background: linear-gradient(90deg, #b91c1c, #f85149); }

  /* Confidence banner */
  .conf-banner {
      border-radius: 8px; padding: 0.55rem 1rem;
      font-family: monospace; font-size: 0.85rem;
      font-weight: 600; letter-spacing: 0.04em;
  }
  .conf-very-high { background:#1a2e1a; border:1px solid #3fb950; color:#3fb950; }
  .conf-high      { background:#162032; border:1px solid #58a6ff; color:#58a6ff; }
  .conf-moderate  { background:#2d2213; border:1px solid #d29922; color:#d29922; }
  .conf-low       { background:#3d1f1f; border:1px solid #f85149; color:#f85149; }

  /* Divider */
  .divider { border:none; border-top:1px solid #30363d; margin: 1.2rem 0; }

  /* Section heading */
  .sec-heading {
      font-size: 0.7rem; letter-spacing: 0.14em; text-transform: uppercase;
      color: #8b949e; margin-bottom: 0.8rem; font-family: monospace;
  }

  /* Reset button */
  div[data-testid="stButton"] > button {
      width: 100%;
      background: #21262d;
      border: 1px solid #30363d;
      color: #e6edf3;
      border-radius: 8px;
      padding: 0.6rem;
      font-family: monospace;
      font-size: 0.85rem;
      letter-spacing: 0.06em;
      transition: background 0.2s, border-color 0.2s;
  }
  div[data-testid="stButton"] > button:hover {
      background: #30363d;
      border-color: #58a6ff;
      color: #58a6ff;
  }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 1.8rem 0 0.6rem">
  <div style="font-size:2rem; margin-bottom:4px">🧠</div>
  <h1 style="margin:0; font-size:1.6rem; font-family:monospace; letter-spacing:0.06em; color:#e6edf3;">
    Brain Tumor Detection
  </h1>
  <p style="margin:6px 0 0; font-size:0.82rem; color:#8b949e; font-family:monospace;">
    EfficientNet · Grad-CAM · MRI Analysis
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ── Upload ───────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload an MRI image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    input_array = preprocess_image(image)

    # ── Run inference ────────────────────────────────────────────────────────
    preds = model.predict(input_array, verbose=0)[0]
    class_idx = int(np.argmax(preds))
    class_label = CLASS_NAMES[class_idx]
    confidence = preds[class_idx] * 100

    # ── Generate Grad-CAM ────────────────────────────────────────────────────
    gradcam_img = generate_gradcam(input_array, image)

    # Convert both to same fixed size for display parity
    display_size = (400, 400)
    orig_display = image.resize(display_size)
    gradcam_pil = Image.fromarray(gradcam_img).resize(display_size)

    # ── 1. Visualization (FIRST) ─────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-heading">🔬 Grad-CAM Visualization</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image(orig_display, caption="Original MRI", use_container_width=True)
    with col2:
        st.image(gradcam_pil, caption="Grad-CAM Activation", use_container_width=True)

    st.markdown(
        '<p style="font-size:0.72rem;color:#8b949e;font-family:monospace;margin-top:6px;">'
        'Warm regions (red/orange) indicate areas of highest model attention.</p>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── 2. Prediction result ─────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-heading">🩺 Prediction Result</div>', unsafe_allow_html=True)

    verdict_class = "verdict-tumor" if class_label != "notumor" else "verdict-notumor"
    icon = "🔴" if class_label != "notumor" else "🟢"
    st.markdown(
        f'<div class="{verdict_class}">{icon}&nbsp;{class_label.upper()}</div>'
        f'<div class="conf-text">Confidence: {confidence:.2f}%</div>',
        unsafe_allow_html=True
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="sec-heading">📊 Class Probabilities</div>', unsafe_allow_html=True)

    for i, prob in enumerate(preds):
        pct = prob * 100
        is_top = (i == class_idx)
        fill_class = "high" if (is_top and class_label != "notumor") else ""
        st.markdown(f"""
        <div class="prob-row">
          <div class="prob-label">
            <span>{'▶ ' if is_top else ''}{CLASS_NAMES[i]}</span>
            <span>{pct:.2f}%</span>
          </div>
          <div class="prob-bar-bg">
            <div class="prob-bar-fill {fill_class}" style="width:{pct:.1f}%"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Confidence banner
    if confidence > 90:
        banner = '<div class="conf-banner conf-very-high">🔥 Very High Confidence — Model is highly certain.</div>'
    elif confidence > 75:
        banner = '<div class="conf-banner conf-high">✅ High Confidence — Result is reliable.</div>'
    elif confidence > 60:
        banner = '<div class="conf-banner conf-moderate">⚠️ Moderate Confidence — Consider clinical review.</div>'
    else:
        banner = '<div class="conf-banner conf-low">❗ Low Confidence — Manual review strongly advised.</div>'

    st.markdown(banner, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── 3. Analyze another image ─────────────────────────────────────────────
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
    if st.button("🔄  Analyze Another Image"):
        st.rerun()

else:
    # Upload placeholder
    st.markdown("""
    <div style="
        border: 1px dashed #30363d;
        border-radius: 12px;
        padding: 2.8rem 2rem;
        text-align: center;
        color: #8b949e;
        font-family: monospace;
        font-size: 0.82rem;
        letter-spacing: 0.05em;
        background: #161b22;
        margin-top: 0.5rem;
    ">
        <div style="font-size:2rem;margin-bottom:10px">🩻</div>
        Upload a PNG / JPG / JPEG MRI scan to begin analysis.<br>
        <span style="color:#30363d;">Supports axial, coronal & sagittal views.</span>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center;margin-top:2rem;font-family:monospace;
    font-size:0.68rem;color:#484f58;letter-spacing:0.08em;">
  FOR RESEARCH USE ONLY · NOT A MEDICAL DEVICE
</div>
""", unsafe_allow_html=True)