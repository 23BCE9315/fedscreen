import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import streamlit as st

from PIL import Image

from fedscreen_model import FedScreenMultiTaskModel


# ==============================================================================
# Streamlit Configuration
# ==============================================================================

st.set_page_config(
    page_title="FedScreen AI Dashboard",
    page_icon="🔬",
    layout="wide"
)


# ==============================================================================
# Device
# ==============================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==============================================================================
# Page Styling
# ==============================================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #666;
        margin-bottom: 25px;
    }

    .debug-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f5f5f5;
        border: 1px solid #ddd;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# Model Loading
# ==============================================================================

@st.cache_resource
def load_model():

    model = FedScreenMultiTaskModel().to(DEVICE)

    app_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    checkpoint_candidates = [
        os.path.join(
            app_dir,
            "fedscreen_fedprox_global_model.pth"
        ),
        os.path.join(
            app_dir,
            "models",
            "fedscreen_fedprox_global_model.pth"
        )
    ]

    checkpoint_path = next(
        (
            path
            for path in checkpoint_candidates
            if os.path.isfile(path)
        ),
        None
    )

    if checkpoint_path is None:
        raise FileNotFoundError(
            "FedProx model checkpoint was not found."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint,
        strict=True
    )

    model.eval()

    return model, checkpoint_path


# ==============================================================================
# Image Preprocessing
# ==============================================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        (128, 128),
        Image.Resampling.BILINEAR
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array /= 255.0

    tensor = torch.from_numpy(
        image_array
    )

    tensor = tensor.permute(
        2,
        0,
        1
    )

    tensor = tensor.unsqueeze(0)

    return tensor.to(DEVICE)


# ==============================================================================
# Grad-CAM++
# ==============================================================================

class GradCAMPlusPlus:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self._save_activations
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self._save_gradients
            )
        )

    def _save_activations(
        self,
        module,
        inputs,
        output
    ):

        self.activations = output

    def _save_gradients(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]

    def generate(
        self,
        input_tensor,
        task,
        target_class
    ):

        self.model.eval()

        self.model.zero_grad(
            set_to_none=True
        )

        predictions = self.model(
            input_tensor
        )

        logits = predictions[task]

        score = logits[
            0,
            target_class
        ]

        score.backward(
            retain_graph=False
        )

        if self.gradients is None:
            raise RuntimeError(
                "Gradients were not captured."
            )

        if self.activations is None:
            raise RuntimeError(
                "Activations were not captured."
            )

        gradients = self.gradients[0]
        activations = self.activations[0]

        grad_squared = gradients ** 2
        grad_cubed = gradients ** 3

        sum_activations = torch.sum(
            activations,
            dim=(1, 2),
            keepdim=True
        )

        denominator = (
            2.0 * grad_squared
            + sum_activations * grad_cubed
            + 1e-7
        )

        alpha = (
            grad_squared /
            denominator
        )

        alpha = torch.where(
            gradients != 0,
            alpha,
            torch.zeros_like(alpha)
        )

        weights = torch.sum(
            alpha * F.relu(gradients),
            dim=(1, 2),
            keepdim=True
        )

        cam = torch.sum(
            weights * activations,
            dim=0
        )

        cam = F.relu(cam)

        cam = (
            cam.detach()
            .cpu()
            .numpy()
        )

        cam = cv2.resize(
            cam,
            (128, 128),
            interpolation=cv2.INTER_LINEAR
        )

        cam_min = cam.min()
        cam_max = cam.max()

        cam -= cam_min

        if cam_max - cam_min > 1e-8:
            cam /= (
                cam_max - cam_min
            )

        return cam.astype(
            np.float32
        )

    def close(self):

        self.forward_handle.remove()
        self.backward_handle.remove()


# ==============================================================================
# Grad-CAM Overlay
# ==============================================================================

def create_overlay(
    image,
    heatmap
):

    image = (
        image
        .convert("RGB")
        .resize(
            (128, 128),
            Image.Resampling.BILINEAR
        )
    )

    image_np = np.asarray(
        image,
        dtype=np.uint8
    )

    heatmap_uint8 = np.uint8(
        heatmap * 255
    )

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        image_np,
        0.60,
        heatmap_color,
        0.40,
        0
    )

    return overlay


# ==============================================================================
# Utility Functions
# ==============================================================================

TASK_NAMES = {
    "malaria": "Malaria",
    "anemia": "Anemia",
    "leukemia": "Leukemia"
}


def get_prediction_details(logits):

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    probabilities = (
        probabilities[0]
        .detach()
        .cpu()
        .numpy()
    )

    predicted_class = int(
        np.argmax(probabilities)
    )

    confidence = (
        float(
            probabilities[predicted_class]
        ) * 100
    )

    return (
        probabilities,
        predicted_class,
        confidence
    )


# ==============================================================================
# Header
# ==============================================================================

st.markdown(
    '<div class="main-title">🔬 FedScreen AI Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Privacy-Preserving Multi-Disease Blood-Smear Screening
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    Upload a peripheral blood-smear image to inspect the
    current FedScreen model predictions.
    """
)


# ==============================================================================
# Sidebar
# ==============================================================================

st.sidebar.header(
    "⚙️ Control Panel"
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold (%)",
    min_value=50,
    max_value=95,
    value=85
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Blood Smear Image",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)

st.sidebar.markdown("---")

st.sidebar.write(
    f"**Device:** {DEVICE}"
)

st.sidebar.write(
    "**Input size:** 128 × 128"
)

st.sidebar.write(
    "**Architecture:** EfficientNet-B0 + CBAM"
)


# ==============================================================================
# Load Model
# ==============================================================================

try:

    model, checkpoint_path = load_model()

except Exception as exc:

    st.error(
        f"Unable to load model: {exc}"
    )

    st.stop()


# Show loaded checkpoint in sidebar

st.sidebar.write(
    f"**Checkpoint:** `{os.path.basename(checkpoint_path)}`"
)


# ==============================================================================
# Main Application
# ==============================================================================

if uploaded_file is not None:

    # ==========================================================================
    # Read Image
    # ==========================================================================

    original_image = Image.open(
        uploaded_file
    ).convert("RGB")


    # ==========================================================================
    # Preprocess
    # ==========================================================================

    input_tensor = preprocess_image(
        original_image
    )


    # ==========================================================================
    # Image Display
    # ==========================================================================

    col1, col2 = st.columns(
        [1, 1]
    )

    with col1:

        st.subheader(
            "🩸 Input Blood-Smear Image"
        )

        st.image(
            original_image,
            use_container_width=True
        )

        st.caption(
            f"Original image size: "
            f"{original_image.width} × "
            f"{original_image.height}"
        )


    # ==========================================================================
    # Model Prediction
    # ==========================================================================

    # IMPORTANT:
    # We calculate prediction once and keep the raw logits.
    # Grad-CAM is calculated separately later.

    with torch.no_grad():

        predictions = model(
            input_tensor
        )


    # ==========================================================================
    # Extract Predictions
    # ==========================================================================

    prediction_data = {}

    for task_name, logits in predictions.items():

        probs, predicted_class, confidence = (
            get_prediction_details(
                logits
            )
        )

        raw_logits = (
            logits[0]
            .detach()
            .cpu()
            .numpy()
        )

        prediction_data[task_name] = {

            "probabilities": probs,

            "predicted_class":
                predicted_class,

            "confidence":
                confidence,

            "raw_logits":
                raw_logits
        }


    # ==========================================================================
    # Results
    # ==========================================================================

    with col2:

        st.subheader(
            "🧠 AI Screening Results"
        )

        for task_name in [
            "malaria",
            "anemia",
            "leukemia"
        ]:

            data = prediction_data[
                task_name
            ]

            probs = data[
                "probabilities"
            ]

            predicted_class = data[
                "predicted_class"
            ]

            confidence = data[
                "confidence"
            ]

            display_name = TASK_NAMES[
                task_name
            ]

            st.markdown(
                f"### {display_name}"
            )

            st.write(
                f"Predicted class: "
                f"**Class {predicted_class}**"
            )

            st.write(
                f"Prediction confidence: "
                f"**{confidence:.2f}%**"
            )

            st.progress(
                float(
                    confidence / 100
                )
            )

            st.write(
                f"Class 0 probability: "
                f"**{probs[0] * 100:.2f}%**"
            )

            st.write(
                f"Class 1 probability: "
                f"**{probs[1] * 100:.2f}%**"
            )

            if confidence < confidence_threshold:

                st.warning(
                    "Confidence is below the configured "
                    "threshold. Manual review recommended."
                )

            else:

                st.success(
                    "Prediction is above the configured "
                    "confidence threshold."
                )

            st.markdown("---")


    # ==========================================================================
    # RAW LOGIT DIAGNOSTICS
    # ==========================================================================

    st.subheader(
        "🧪 Raw Model Output Diagnostics"
    )

    st.caption(
        """
        These are the actual two output values produced by each
        classification head before Softmax. We use these values
        to determine whether the model is responding to the image.
        """
    )

    for task_name in [
        "malaria",
        "anemia",
        "leukemia"
    ]:

        data = prediction_data[
            task_name
        ]

        logits = data[
            "raw_logits"
        ]

        difference = abs(
            float(logits[0])
            -
            float(logits[1])
        )

        st.markdown(
            f"### {TASK_NAMES[task_name]}"
        )

        log_col1, log_col2, log_col3 = st.columns(
            3
        )

        with log_col1:

            st.metric(
                "Class 0 Logit",
                f"{logits[0]:.8f}"
            )

        with log_col2:

            st.metric(
                "Class 1 Logit",
                f"{logits[1]:.8f}"
            )

        with log_col3:

            st.metric(
                "Absolute Difference",
                f"{difference:.8f}"
            )


    # ==========================================================================
    # Input Diagnostic Information
    # ==========================================================================

    st.subheader(
        "🔍 Input Diagnostic Information"
    )

    st.caption(
        """
        These values tell us whether the uploaded image is actually
        different after preprocessing.
        """
    )

    debug_col1, debug_col2, debug_col3, debug_col4 = st.columns(
        4
    )

    with debug_col1:

        st.metric(
            "Tensor Shape",
            str(
                tuple(
                    input_tensor.shape
                )
            )
        )

    with debug_col2:

        st.metric(
            "Minimum",
            f"{float(input_tensor.min()):.6f}"
        )

    with debug_col3:

        st.metric(
            "Maximum",
            f"{float(input_tensor.max()):.6f}"
        )

    with debug_col4:

        st.metric(
            "Mean",
            f"{float(input_tensor.mean()):.6f}"
        )

    st.write(
        f"**Standard deviation:** "
        f"{float(input_tensor.std()):.6f}"
    )


    # ==========================================================================
    # Image Fingerprint
    # ==========================================================================

    st.subheader(
        "🧬 Image Fingerprint"
    )

    image_fingerprint = (
        input_tensor
        .detach()
        .cpu()
        .numpy()
    )

    image_fingerprint = np.round(
        image_fingerprint,
        decimals=6
    )

    fingerprint_value = float(
        np.sum(
            image_fingerprint
        )
    )

    st.write(
        f"**Input tensor sum:** "
        f"`{fingerprint_value:.8f}`"
    )

    st.caption(
        """
        If this value changes between different uploaded images,
        the images are reaching the neural network as different tensors.
        """ 
    )


    # ==========================================================================
    # Grad-CAM++
    # ==========================================================================

    st.subheader(
        "🔥 Grad-CAM++ Explainability"
    )

    cam_col1, cam_col2 = st.columns(
        [1, 2]
    )

    with cam_col1:

        selected_task = st.selectbox(
            "Disease",
            [
                "malaria",
                "anemia",
                "leukemia"
            ]
        )

        selected_class = st.selectbox(
            "Class to explain",
            [
                0,
                1
            ],
            format_func=lambda x:
                (
                    "Class 0"
                    if x == 0
                    else "Class 1"
                )
        )


    with cam_col2:

        target_layer = model.backbone[-1]

        cam_generator = GradCAMPlusPlus(
            model,
            target_layer
        )

        try:

            heatmap = cam_generator.generate(
                input_tensor,
                selected_task,
                selected_class
            )

            overlay = create_overlay(
                original_image,
                heatmap
            )

            st.image(
                overlay,
                caption=(
                    "Grad-CAM++ — "
                    f"{TASK_NAMES[selected_task]} — "
                    f"Class {selected_class}"
                ),
                use_container_width=True
            )

        except Exception as exc:

            st.error(
                f"Grad-CAM++ generation failed: {exc}"
            )

        finally:

            cam_generator.close()


    # ==========================================================================
    # Prediction Summary
    # ==========================================================================

    st.subheader(
        "📊 Prediction Summary"
    )

    summary_header = st.columns(
        4
    )

    summary_header[0].markdown(
        "**Disease**"
    )

    summary_header[1].markdown(
        "**Class 0**"
    )

    summary_header[2].markdown(
        "**Class 1**"
    )

    summary_header[3].markdown(
        "**Predicted**"
    )


    for task_name in [
        "malaria",
        "anemia",
        "leukemia"
    ]:

        data = prediction_data[
            task_name
        ]

        probs = data[
            "probabilities"
        ]

        predicted_class = data[
            "predicted_class"
        ]

        row = st.columns(
            4
        )

        row[0].write(
            TASK_NAMES[
                task_name
            ]
        )

        row[1].write(
            f"{probs[0] * 100:.2f}%"
        )

        row[2].write(
            f"{probs[1] * 100:.2f}%"
        )

        row[3].write(
            f"Class {predicted_class}"
        )


    # ==========================================================================
    # Important Disclaimer
    # ==========================================================================

    st.warning(
        """
        **Important:** FedScreen is a research/prototype screening system.
        Its predictions should not be used as a standalone medical diagnosis.
        Results require appropriate clinical/laboratory verification.
        """
    )


    # ==========================================================================
    # Current Inference Pipeline
    # ==========================================================================

    with st.expander(
        "ℹ️ Current inference pipeline"
    ):

        st.write(
            """
            Current deployment flow:

            Uploaded image
            ↓
            RGB conversion
            ↓
            Resize to 128 × 128
            ↓
            Pixel scaling to 0–1
            ↓
            EfficientNet-B0 backbone
            ↓
            CBAM disease heads
            ↓
            Malaria / Anemia / Leukemia predictions
            ↓
            Grad-CAM++ explanation
            """
        )

        st.info(
            """
            The project documentation describes additional training-time
            preprocessing such as Watershed segmentation and Macenko/YUV
            normalization.

            Those exact transformations are not being added blindly here.
            We need to verify the actual training implementation before
            changing inference preprocessing.
            """
        )


else:

    # ==========================================================================
    # Empty State
    # ==========================================================================

    st.info(
        "👈 Upload a blood-smear image from the sidebar to begin."
    )

    st.markdown(
        """
        ### What this dashboard provides

        | Feature | Description |
        |---|---|
        | 🩸 Image Input | Upload a blood-smear image |
        | 🦠 Malaria | Binary model prediction |
        | 🩸 Anemia | Binary model prediction |
        | 🔬 Leukemia | Binary model prediction |
        | 🔥 Grad-CAM++ | Visual explanation |
        | 📊 Confidence | Class probabilities |
        | 🧪 Raw Logits | Model output diagnostics |
        | 🔍 Diagnostics | Input tensor statistics |

        **Model input:** 128 × 128 RGB image
        """
    )