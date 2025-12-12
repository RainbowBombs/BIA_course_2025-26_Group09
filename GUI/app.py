import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import time
import torch.nn.functional as F
import warnings

# --- Block system warning ---
warnings.filterwarnings("ignore")

# ==========================================
# 1. Configuration
# ==========================================
MODEL_PATH = 'BreaKHis Analysis Network_best.pth'
CLASSES = {0: 'Benign', 1: 'Malignant'}

# Input dimensions
IMG_HEIGHT = 700
IMG_WIDTH = 460

# Normalization parameters
NORM_MEAN = [0.756, 0.589, 0.742]
NORM_STD = [0.143, 0.201, 0.116]


def get_transform():
    """
    Image preprocessing
    """
    return transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


@st.cache_resource
def load_model():
    """
    Load model
    """
    if not os.path.exists(MODEL_PATH):
        return None

    try:
        # 1. Rebuild Model
        model = models.resnet18(weights=None)

        # 2. Modify FC layer
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)

        # 3. Load Weights
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

        # 4. Set to Eval mode
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def predict(model, image):
    """
    Inference Function
    """
    # 1. Preprocess
    transform = get_transform()
    img_tensor = transform(image).unsqueeze(0)

    # 2. Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)

    # 3. Parse Result
    malignant_prob = probabilities[0][1].item()

    prob, predicted_class = torch.max(probabilities, 1)
    label_idx = predicted_class.item()

    return label_idx, malignant_prob


# ==========================================
# 2. Streamlit UI
# ==========================================

st.set_page_config(
    page_title="Breast Cancer AI Diagnosis",
    page_icon="🔬",
    layout="wide"
)

# --- Sidebar ---
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Malignancy Threshold", 0.0, 1.0, 0.5, 0.05)
st.sidebar.markdown("---")

# Load Model
model = load_model()

if model is None:
    st.sidebar.warning(f"⚠️ Model file `{MODEL_PATH}` not found.")
    st.sidebar.info("Status: Demo Mode (Random Data)")
else:
    st.sidebar.success("✅ Model Loaded (BreaKHis Analysis Network)")
    st.sidebar.info("Status: AI Inference Mode")

# --- Main Page ---
st.title("🔬 Breast Cancer Pathology AI Diagnosis")
st.markdown(f"**Model Architecture:** BreaKHis Analysis Network | **Input Size:** {IMG_HEIGHT}x{IMG_WIDTH}")
st.markdown("---")

uploaded_file = st.file_uploader("Upload Pathology Slide (jpg, png, tif)", type=["jpg", "jpeg", "png", "tif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    col1, col2 = st.columns([1, 1])

    # Left Column: Original Image
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    # Right Column: Report
    with col2:
        st.subheader("Diagnosis Report")
        st.write("")

        analyze_btn = st.button("Analyze Image", type="primary", use_container_width=True)

        if analyze_btn:
            with st.spinner('Analyzing cell morphology...'):
                time.sleep(0.5)

                if model:
                    # --- Real Inference ---
                    label_idx, malignant_score = predict(model, image)
                else:
                    # --- Demo Mode ---
                    import random

                    malignant_score = random.random()
                    time.sleep(1)

                # --- Display Results ---
                st.markdown("---")

                # Logic based on threshold
                is_malignant = malignant_score > confidence_threshold
                result_text = "Malignant" if is_malignant else "Benign"

                if is_malignant:
                    st.error(f"## Prediction: {result_text}")
                    st.markdown(f"### ⚠️ Risk Level: High")
                else:
                    st.success(f"## Prediction: {result_text}")
                    st.markdown(f"### ✅ Risk Level: Low")

                # Progress Bar
                st.write(f"Malignancy Probability: **{malignant_score * 100:.2f}%**")
                st.progress(malignant_score)

                # Detailed Data
                with st.expander("View Detailed Data"):
                    st.json({
                        "Predicted_Class": result_text,
                        "Malignancy_Score": float(f"{malignant_score:.4f}"),
                        "Threshold_Used": confidence_threshold,
                        "Image_Size": image.size,
                        "Model": "ResNet18" if model else "Demo_Random"
                    })
