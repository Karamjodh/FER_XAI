"""
compute_face_concentration.py
─────────────────────────────
Computes face-region attribution concentration (%) for LIME and SHAP
across ResNet-50 and EfficientNet-B0 on FER2013 and RAF-DB.

HOW IT WORKS
────────────
1. Loads 20 random test images per dataset.
2. Runs LIME and SHAP on each image for each model.
3. Detects the face bounding box using OpenCV Haar cascade.
4. Computes what % of total positive attribution mass falls inside
   the face bounding box.
5. Averages over 20 samples and prints the final table.

USAGE
─────
1. Copy this file into your project root (same level as Config.py).
2. Install dependency if needed:
       pip install opencv-python
3. Run:
       python compute_face_concentration.py

OUTPUT
──────
Prints a table like:
    Method / Model          FER2013    RAF-DB
    LIME  - ResNet-50       78.4       83.1
    LIME  - EfficientNet-B0 76.9       81.7
    SHAP  - ResNet-50       63.2       69.8
    SHAP  - EfficientNet-B0 61.5       67.4
"""

import os
import random
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# ── your existing project imports ─────────────────────────────────
from Config  import NUM_CLASSES
from Models  import build_resnet50, build_efficientnet_b0
from Datasets import get_dataloaders

# ── LIME / SHAP ───────────────────────────────────────────────────
from lime import lime_image
from skimage.segmentation import quickshift
import shap

# ── OpenCV for face detection ─────────────────────────────────────
import cv2

# ─────────────────────────────────────────────────────────────────
# CONFIG  — edit these paths to match your project
# ─────────────────────────────────────────────────────────────────
RESNET_CKPT_FER    = "outputs/checkpoints/resnet50_fer2013_best.pth"
RESNET_CKPT_RAF    = "outputs/checkpoints/resnet50_rafdb_best.pth"
EFFNET_CKPT_FER    = "outputs/checkpoints/efficientnet_b0_fer2013_best.pth"
EFFNET_CKPT_RAF    = "outputs/checkpoints/efficientnet_b0_rafdb_best.pth"

FER2013_TEST_DIR   = "data/fer2013/test"   # folder with class subfolders
RAFDB_TEST_DIR     = "data/rafdb/test"     # folder with class subfolders

N_SAMPLES          = 20      # samples per model-dataset combo
N_LIME_SAMPLES     = 500     # LIME perturbation samples (fast but decent)
N_SHAP_BACKGROUND  = 50      # SHAP background samples

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────
# HAAR CASCADE  (ships with OpenCV, no download needed)
# ─────────────────────────────────────────────────────────────────
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def get_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])

def load_model(model_fn, ckpt_path):
    model = model_fn(num_classes=7, pretrained=False)
    state = torch.load(ckpt_path, map_location=DEVICE)
    # handle DataParallel / plain state_dict
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval().to(DEVICE)
    return model

def collect_test_images(test_dir, n=20):
    """Randomly collect n image paths from a test directory."""
    all_paths = []
    for cls in os.listdir(test_dir):
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                all_paths.append(os.path.join(cls_dir, fname))
    random.shuffle(all_paths)
    return all_paths[:n]

def detect_face_mask(img_np_224):
    """
    img_np_224 : HxWx3 uint8 numpy array (224x224)
    Returns a binary mask (224x224) where 1 = inside face bounding box.
    Falls back to central 60% crop if no face is detected.
    """
    gray = cv2.cvtColor(img_np_224, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
    )
    mask = np.zeros((224, 224), dtype=np.float32)
    if len(faces) > 0:
        # use the largest detected face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        # expand bbox by 10% for safety margin
        pad = int(0.10 * max(w, h))
        x1 = max(0,   x - pad)
        y1 = max(0,   y - pad)
        x2 = min(224, x + w + pad)
        y2 = min(224, y + h + pad)
        mask[y1:y2, x1:x2] = 1.0
    else:
        # fallback: central 60% region
        margin = int(224 * 0.20)
        mask[margin:224-margin, margin:224-margin] = 1.0
    return mask

def concentration_score(attr_map, face_mask):
    """
    attr_map  : HxW numpy array of attribution values (can be negative)
    face_mask : HxW binary mask (1 = face region)
    Returns % of positive attribution mass inside face region.
    """
    pos = np.maximum(attr_map, 0)
    total_pos = pos.sum()
    if total_pos < 1e-8:
        return 0.0
    inside = (pos * face_mask).sum()
    return float(inside / total_pos) * 100.0


# ─────────────────────────────────────────────────────────────────
# LIME CONCENTRATION
# ─────────────────────────────────────────────────────────────────

def lime_concentration(model, img_path, transform, device):
    img_pil = Image.open(img_path).convert("RGB").resize((224, 224))
    img_np  = np.array(img_pil, dtype=np.uint8)          # 224x224x3

    def predict_fn(images):
        batch = torch.stack([
            transform(Image.fromarray(im.astype(np.uint8)))
            for im in images
        ]).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs  = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    explainer   = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_np,
        predict_fn,
        top_labels=1,
        num_samples=N_LIME_SAMPLES,
        segmentation_fn=lambda x: quickshift(x, kernel_size=4,
                                             max_dist=200, ratio=0.2),
    )
    pred_label  = explanation.top_labels[0]
    seg, weights = explanation.get_image_and_mask(
        pred_label,
        positive_only=True,
        num_features=10,
        hide_rest=False,
    )
    # weights map: superpixel -> weight
    sp_weights  = dict(explanation.local_exp[pred_label])
    segments    = explanation.segments          # HxW int array
    attr_map    = np.zeros((224, 224), dtype=np.float32)
    for sp_id, w in sp_weights.items():
        attr_map[segments == sp_id] = w

    face_mask   = detect_face_mask(img_np)
    return concentration_score(attr_map, face_mask)


# ─────────────────────────────────────────────────────────────────
# SHAP CONCENTRATION
# ─────────────────────────────────────────────────────────────────

def shap_concentration(model, img_path, background_paths,
                       transform, device):
    img_pil = Image.open(img_path).convert("RGB").resize((224, 224))
    img_np  = np.array(img_pil, dtype=np.uint8)

    # build background tensor
    bg_tensors = torch.stack([
        transform(Image.open(p).convert("RGB").resize((224, 224)))
        for p in background_paths
    ]).to(device)

    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    explainer  = shap.DeepExplainer(model, bg_tensors)
    shap_vals  = explainer.shap_values(img_tensor)   # list[7] of 1xCxHxW

    with torch.no_grad():
        pred = model(img_tensor).argmax(dim=1).item()

    # aggregate channels → HxW
    sv         = shap_vals[pred][0]           # CxHxW
    attr_map   = sv.sum(axis=0)               # HxW  (sum over channels)

    face_mask  = detect_face_mask(img_np)
    return concentration_score(attr_map, face_mask)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def evaluate(model_name, model, test_dir, dataset_name,
             transform, device, background_paths=None):
    images = collect_test_images(test_dir, n=N_SAMPLES)
    lime_scores, shap_scores = [], []

    for i, img_path in enumerate(images):
        print(f"  [{i+1:02d}/{N_SAMPLES}] {os.path.basename(img_path)}")
        try:
            ls = lime_concentration(model, img_path, transform, device)
            lime_scores.append(ls)
        except Exception as e:
            print(f"    LIME error: {e}")

        try:
            bg = background_paths if background_paths else images
            ss = shap_concentration(model, img_path, bg[:N_SHAP_BACKGROUND],
                                    transform, device)
            shap_scores.append(ss)
        except Exception as e:
            print(f"    SHAP error: {e}")

    lime_mean = np.mean(lime_scores) if lime_scores else 0.0
    shap_mean = np.mean(shap_scores) if shap_scores else 0.0
    print(f"  → LIME {dataset_name}: {lime_mean:.1f}%  "
          f"SHAP {dataset_name}: {shap_mean:.1f}%")
    return lime_mean, shap_mean


if __name__ == "__main__":
    transform = get_transform()

    configs = [
        ("ResNet-50",        build_resnet50,        RESNET_CKPT_FER, RESNET_CKPT_RAF),
        ("EfficientNet-B0",  build_efficientnet_b0, EFFNET_CKPT_FER, EFFNET_CKPT_RAF),
    ]

    results = {}   # key: (method, model_name, dataset)  value: float

    for model_name, model_fn, ckpt_fer, ckpt_raf in configs:
        for dataset_name, ckpt, test_dir in [
            ("FER2013", ckpt_fer, FER2013_TEST_DIR),
            ("RAF-DB",  ckpt_raf, RAFDB_TEST_DIR),
        ]:
            print(f"\n{'='*55}")
            print(f"  {model_name} on {dataset_name}")
            print(f"{'='*55}")
            model       = load_model(model_fn, ckpt)
            bg_paths    = collect_test_images(test_dir, n=N_SHAP_BACKGROUND)
            lime_m, shap_m = evaluate(
                model_name, model, test_dir, dataset_name,
                transform, DEVICE, background_paths=bg_paths
            )
            results[("LIME", model_name, dataset_name)] = lime_m
            results[("SHAP", model_name, dataset_name)] = shap_m

    # ── Print final table ─────────────────────────────────────────
    print("\n\n" + "="*55)
    print("  FINAL FACE-REGION CONCENTRATION TABLE")
    print("="*55)
    print(f"  {'Method / Model':<30} {'FER2013':>8} {'RAF-DB':>8}")
    print("-"*55)
    for method in ["LIME", "SHAP"]:
        for mname in ["ResNet-50", "EfficientNet-B0"]:
            fer = results.get((method, mname, "FER2013"), 0.0)
            raf = results.get((method, mname, "RAF-DB"),  0.0)
            print(f"  {method} --- {mname:<26} {fer:>7.1f}%  {raf:>7.1f}%")
    print("="*55)
    print("\nPaste these numbers into your LaTeX table.")