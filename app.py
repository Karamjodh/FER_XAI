"""
app.py — FER XAI Live Showcase Backend
Loads your actual trained ResNet-50 / EfficientNet-B0 checkpoints and serves
real-time predictions + real LIME and SHAP explanations over a local Flask API.

Run with:  python app.py
Then open: http://localhost:5000  (serves the frontend automatically)
"""

import io
import base64
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lime import lime_image
from skimage.segmentation import quickshift
import shap

# ──────────────────────────────────────────────────────────────────────────
# CONFIG — adjust these paths to match your repo's outputs/checkpoints/
# ──────────────────────────────────────────────────────────────────────────
CHECKPOINTS = {
    "resnet50_rafdb": "outputs/checkpoints/resnet50_rafdb_best.pth",
    "efficientnet_b0_fer2013": "outputs/checkpoints/efficientnet_b0_fer2013_best.pth",
}
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 224

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ──────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS — must match your Models.py exactly
# ──────────────────────────────────────────────────────────────────────────
def set_inplace_false(module):
    for m in module.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False

class FERResNet50(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        in_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=False),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )
        set_inplace_false(self)

    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(feats)


class FEREfficientNetB0(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=False),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )
        set_inplace_false(self)

    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(feats)


def build_resnet50():
    return FERResNet50()

def build_efficientnet_b0():
    return FEREfficientNetB0()

MODEL_BUILDERS = {
    "resnet50_rafdb": build_resnet50,
    "efficientnet_b0_fer2013": build_efficientnet_b0,
}

# ──────────────────────────────────────────────────────────────────────────
# LOAD MODELS AT STARTUP
# ──────────────────────────────────────────────────────────────────────────
loaded_models = {}

def load_model(key):
    if key in loaded_models:
        return loaded_models[key]
    print(f"[startup] Loading {key} from {CHECKPOINTS[key]} ...")
    model = MODEL_BUILDERS[key]()
    try:
        state = torch.load(CHECKPOINTS[key], map_location=DEVICE, weights_only=False)
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model_state_dict" in state:
                state = state["model_state_dict"]

        try:
            model.load_state_dict(state)
            print(f"[startup] {key} loaded successfully (strict match).")
        except RuntimeError as e:
            # Try non-strict load and report exactly what didn't match,
            # so a naming mismatch is diagnosable in seconds instead of
            # scrolling a 300-line traceback.
            missing, unexpected = model.load_state_dict(state, strict=False)
            ckpt_keys = set(state.keys())
            model_keys = set(model.state_dict().keys())
            print(f"[WARNING] Non-strict load for {key}.")
            print(f"  Checkpoint has {len(ckpt_keys)} keys, model expects {len(model_keys)} keys.")
            print(f"  Sample checkpoint keys: {list(ckpt_keys)[:5]}")
            print(f"  Sample model keys:      {list(model_keys)[:5]}")
            if missing:
                print(f"  Missing in checkpoint ({len(missing)}): {missing[:5]} ...")
            if unexpected:
                print(f"  Unexpected in checkpoint ({len(unexpected)}): {unexpected[:5]} ...")
            if len(missing) == 0 and len(unexpected) == 0:
                print(f"[startup] {key} actually loaded fine — ignore above, this was a false alarm.")
            else:
                print(f"[ERROR] {key} weights only PARTIALLY loaded. Predictions will be unreliable. "
                      f"Fix the FERResNet50/FEREfficientNetB0 class in app.py to match your Models.py exactly.")
    except FileNotFoundError:
        print(f"[WARNING] Checkpoint not found at {CHECKPOINTS[key]} — "
              f"using untrained weights for {key}. Update CHECKPOINTS path.")
    model.to(DEVICE)
    model.eval()
    loaded_models[key] = model
    return model

print("=" * 60)
print("FER XAI Live Backend — loading checkpoints")
print("=" * 60)
for k in CHECKPOINTS:
    load_model(k)
print("=" * 60)
print(f"Device: {DEVICE}")
print("All models loaded. Starting Flask server...")
print("=" * 60)

# ──────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────
def decode_base64_image(data_url):
    """Decode a 'data:image/jpeg;base64,...' string into a PIL RGB image."""
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return img

def predict_proba(model, pil_image):
    """Run a single forward pass, return softmax probabilities as numpy array."""
    x = preprocess(pil_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs

def lime_predict_fn(model):
    """Returns a function LIME can call: takes a batch of HWC uint8 images,
    returns an (N, 7) probability array."""
    def predict_fn(images):
        batch = []
        for img in images:
            pil_img = Image.fromarray(img.astype("uint8"))
            batch.append(preprocess(pil_img))
        batch = torch.stack(batch).to(DEVICE)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs
    return predict_fn

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110,
                facecolor="#0a0c10")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

# ──────────────────────────────────────────────────────────────────────────
# LIME EXPLANATION
# ──────────────────────────────────────────────────────────────────────────
lime_explainer = lime_image.LimeImageExplainer()

def run_lime(model, pil_image, top_label, num_samples=300):
    """Reduced num_samples (300 vs the paper's 1000) for near-real-time speed.
    Quality is still representative; full 1000 used in the offline paper results."""
    img_resized = pil_image.resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(img_resized)

    explanation = lime_explainer.explain_instance(
        img_np,
        lime_predict_fn(model),
        top_labels=1,
        hide_color=0,
        num_samples=num_samples,
        segmentation_fn=lambda x: quickshift(x, kernel_size=1.5, max_dist=20, ratio=0.2),
    )

    temp, mask = explanation.get_image_and_mask(
        top_label, positive_only=True, num_features=8, hide_rest=False
    )

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(temp)
    ax.contour(mask, colors="lime", linewidths=1.5)
    ax.axis("off")
    fig.patch.set_facecolor("#0a0c10")
    return fig_to_base64(fig)

# ──────────────────────────────────────────────────────────────────────────
# SHAP EXPLANATION
# ──────────────────────────────────────────────────────────────────────────
shap_explainers = {}
shap_background_cache = {}

def get_shap_explainer(model_key, model):
    if model_key in shap_explainers:
        return shap_explainers[model_key]
    # Lightweight background: 20 random noise-grey images (fast startup).
    # For best fidelity, replace with real stratified training samples per
    # the methodology in the paper (100 samples, 7-class stratified).
    bg = torch.randn(20, 3, IMG_SIZE, IMG_SIZE).to(DEVICE) * 0.2
    explainer = shap.GradientExplainer(model, bg)
    shap_explainers[model_key] = explainer
    return explainer

def run_shap(model, model_key, pil_image, top_label):
    explainer = get_shap_explainer(model_key, model)
    x = preprocess(pil_image).unsqueeze(0).to(DEVICE)
    shap_values, indexes = explainer.shap_values(x, nsamples=50, ranked_outputs=1)

    sv = shap_values[0][0] if isinstance(shap_values, list) else shap_values[..., 0][0]
    sv = np.transpose(sv, (1, 2, 0)).sum(axis=2)  # collapse channels

    img_np = np.array(pil_image.resize((IMG_SIZE, IMG_SIZE))) / 255.0

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(img_np)
    vmax = np.percentile(np.abs(sv), 99) + 1e-8
    ax.imshow(sv, cmap="bwr", alpha=0.55, vmin=-vmax, vmax=vmax)
    ax.axis("off")
    fig.patch.set_facecolor("#0a0c10")
    return fig_to_base64(fig)

# ──────────────────────────────────────────────────────────────────────────
# FLASK APP
# ──────────────────────────────────────────────────────────────────────────
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=STATIC_DIR)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.route("/")
def index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return (
            f"<h2>index.html not found</h2>"
            f"<p>Expected at: <code>{index_path}</code></p>"
            f"<p>Make sure the <code>static/</code> folder sits next to <code>app.py</code> "
            f"(i.e. inside <code>{BASE_DIR}</code>), and that it contains <code>index.html</code>.</p>",
            404,
        )
    return send_from_directory(STATIC_DIR, "index.html")

@app.route("/api/models", methods=["GET"])
def list_models():
    return jsonify({"models": list(CHECKPOINTS.keys())})

@app.route("/api/predict", methods=["POST"])
def predict():
    """Fast endpoint: just classification, called every frame (~10fps)."""
    try:
        payload = request.get_json()
        model_key = payload.get("model", "resnet50_rafdb")
        image_data = payload["image"]

        model = load_model(model_key)
        pil_image = decode_base64_image(image_data)

        t0 = time.time()
        probs = predict_proba(model, pil_image)
        latency_ms = (time.time() - t0) * 1000

        top_idx = int(np.argmax(probs))
        return jsonify({
            "emotion": EMOTIONS[top_idx],
            "emotion_idx": top_idx,
            "probabilities": {EMOTIONS[i]: float(probs[i]) for i in range(7)},
            "latency_ms": round(latency_ms, 1),
            "model": model_key,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/explain", methods=["POST"])
def explain():
    """Slower endpoint: real LIME + SHAP, called on-demand (button press / every N seconds)."""
    try:
        payload = request.get_json()
        model_key = payload.get("model", "resnet50_rafdb")
        image_data = payload["image"]
        methods = payload.get("methods", ["lime", "shap"])

        model = load_model(model_key)
        pil_image = decode_base64_image(image_data)

        probs = predict_proba(model, pil_image)
        top_idx = int(np.argmax(probs))

        result = {
            "emotion": EMOTIONS[top_idx],
            "emotion_idx": top_idx,
            "probabilities": {EMOTIONS[i]: float(probs[i]) for i in range(7)},
            "model": model_key,
        }

        if "lime" in methods:
            t0 = time.time()
            result["lime_image"] = run_lime(model, pil_image, top_idx)
            result["lime_latency_ms"] = round((time.time() - t0) * 1000, 1)

        if "shap" in methods:
            t0 = time.time()
            result["shap_image"] = run_shap(model, model_key, pil_image, top_idx)
            result["shap_latency_ms"] = round((time.time() - t0) * 1000, 1)

        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)