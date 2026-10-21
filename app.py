"""Gradio demo for trained Brain MRI classification checkpoints.

Run from the project root:
    python app.py

The app loads checkpoints produced by notebooks 02, 03, and 04. It does not
train models and does not download pretrained weights during inference.
"""
from __future__ import annotations

import functools
import warnings
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

from src.model_factory import build_model, gradcam_target_layer

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARCHITECTURES = {
    "Custom CNN": "custom_cnn",
    "ResNet-18": "resnet18",
    "ViT-B/16": "vit_b16",
}
CHECKPOINT_PATHS = {
    "custom_cnn": MODELS_DIR / "custom_cnn.pt",
    "resnet18": MODELS_DIR / "resnet18.pt",
    "vit_b16": MODELS_DIR / "vit_b16.pt",
}
DEFAULT_IMG_SIZE = 224
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)

# Prevent accidental loading of massive images in the demo. Typical MRI examples
# are far smaller than this limit.
MAX_IMAGE_PIXELS = 20_000_000


def _safe_torch_load(path: Path) -> dict[str, Any]:
    """Load a weights-only checkpoint when supported by the installed PyTorch."""
    try:
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:  # compatibility with older PyTorch releases
        checkpoint = torch.load(path, map_location=DEVICE)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must be a dictionary: {path}")
    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint does not contain 'model_state_dict': {path}. "
            "Run the corresponding project notebook to generate a compatible checkpoint."
        )
    return checkpoint


def _normalization(checkpoint: dict[str, Any], architecture: str):
    if architecture == "custom_cnn":
        fallback_mean, fallback_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else:
        fallback_mean, fallback_std = DEFAULT_MEAN, DEFAULT_STD

    # Notebook 02/03 use normalization_mean/std. Notebook 04 records
    # imagenet_mean/std. Support both formats for consistency.
    mean = checkpoint.get("normalization_mean", checkpoint.get("imagenet_mean", fallback_mean))
    std = checkpoint.get("normalization_std", checkpoint.get("imagenet_std", fallback_std))
    mean = tuple(float(v) for v in mean)
    std = tuple(float(v) for v in std)
    if len(mean) != 3 or len(std) != 3 or any(v <= 0 for v in std):
        raise ValueError("Checkpoint normalization metadata is invalid.")
    return mean, std


@functools.lru_cache(maxsize=3)
def load_model(architecture: str):
    """Load one fine-tuned model and cache it for subsequent predictions."""
    if architecture not in CHECKPOINT_PATHS:
        raise ValueError(f"Unknown architecture: {architecture}")
    checkpoint_path = CHECKPOINT_PATHS[architecture]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path.relative_to(PROJECT_ROOT)}. "
            "Run the corresponding training notebook first: 02 for Custom CNN, "
            "03 for ResNet-18, or 04 for ViT-B/16."
        )

    checkpoint = _safe_torch_load(checkpoint_path)
    class_names = checkpoint.get("class_names")
    if not isinstance(class_names, (list, tuple)) or len(class_names) < 2:
        raise ValueError(
            f"Checkpoint has missing or invalid class_names metadata: {checkpoint_path.name}"
        )
    class_names = [str(name) for name in class_names]
    img_size = int(checkpoint.get("img_size", DEFAULT_IMG_SIZE))
    if img_size <= 0 or img_size > 1024:
        raise ValueError(f"Unexpected image size in checkpoint: {img_size}")
    mean, std = _normalization(checkpoint, architecture)

    model = build_model(architecture, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()

    preprocessing = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return model, class_names, img_size, preprocessing


def _friendly_class_name(name: str) -> str:
    normalized = name.strip().replace("_", " ").replace("-", " ")
    aliases = {
        "notumor": "No Tumor",
        "no tumor": "No Tumor",
        "no tumour": "No Tumor",
    }
    return aliases.get(normalized.casefold(), normalized.title())


def _to_display_image(image: Image.Image) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError("Please upload a valid image file.")
    if image.width <= 0 or image.height <= 0:
        raise ValueError("The uploaded image is empty.")
    if image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError(
            f"Image is too large ({image.width} × {image.height}). Please upload an image "
            f"with no more than {MAX_IMAGE_PIXELS:,} pixels."
        )
    # Apply EXIF rotation before converting to RGB.
    return ImageOps.exif_transpose(image).convert("RGB")


def _generate_gradcam(model, architecture: str, input_tensor: torch.Tensor,
                      class_index: int, original_image: Image.Image):
    """Create a Grad-CAM overlay for CNN, ResNet-18, or ViT-B/16.

    ViT Grad-CAM uses patch-token activations from the final encoder block and
    reshapes the non-CLS tokens to their square patch grid.
    """
    target_layer = gradcam_target_layer(model, architecture)
    captured: dict[str, torch.Tensor] = {}

    def capture_activations(_module, _inputs, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError("Grad-CAM target layer output is not a tensor.")
        captured["activations"] = output
        if output.requires_grad:
            output.register_hook(lambda grad: captured.__setitem__("gradients", grad))

    hook = target_layer.register_forward_hook(capture_activations)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(input_tensor)
            score = logits[0, int(class_index)]
            score.backward()
        activations = captured.get("activations")
        gradients = captured.get("gradients")
        if activations is None or gradients is None:
            raise RuntimeError("Could not capture gradients for Grad-CAM.")

        if activations.ndim == 4:  # CNN / ResNet: [B, channels, height, width]
            channel_weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.sum(channel_weights * activations, dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(
                cam, size=(original_image.height, original_image.width),
                mode="bilinear", align_corners=False,
            )[0, 0]
        elif activations.ndim == 3:  # ViT: [B, tokens, embedding_dim]
            if activations.shape[1] < 2:
                raise RuntimeError("ViT output does not contain patch tokens.")
            patch_activations = activations[:, 1:, :]  # exclude CLS token
            patch_gradients = gradients[:, 1:, :]
            num_patches = patch_activations.shape[1]
            grid_size = int(round(num_patches ** 0.5))
            if grid_size * grid_size != num_patches:
                raise RuntimeError(f"ViT patch count {num_patches} is not a square grid.")
            token_weights = patch_gradients.mean(dim=1, keepdim=True)
            patch_cam = F.relu((token_weights * patch_activations).sum(dim=-1))
            patch_cam = patch_cam.reshape(1, 1, grid_size, grid_size)
            cam = F.interpolate(
                patch_cam, size=(original_image.height, original_image.width),
                mode="bilinear", align_corners=False,
            )[0, 0]
        else:
            raise RuntimeError(f"Unsupported Grad-CAM activation dimensions: {activations.ndim}")

        cam = cam.detach().float().cpu().numpy()
        cam = np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)
        cam_min, cam_max = float(cam.min()), float(cam.max())
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam, dtype=np.float32)

        # Apply a visible heat map and blend it over the original image.
        import matplotlib.cm as cm
        import matplotlib as mpl
        cmap = mpl.colormaps["jet"](cam)
        # e.g., cmap = plt.cm.get_cmap('jet')
        # cmap = plt.cm.get_cmap('jet')(cam)
        colored = (cmap[..., :3] * 255).astype(np.uint8)
        heatmap_image = Image.fromarray(colored, mode="RGB")
        base = original_image.convert("RGB")
        return Image.blend(base, heatmap_image, alpha=0.38)
    finally:
        hook.remove()
        model.zero_grad(set_to_none=True)


def predict(image: Image.Image | None, model_label: str):
    """Gradio callback; always returns a consistent four-item output tuple."""
    if image is None:
        return None, [], None, "Upload an MRI image to begin."
    if model_label not in ARCHITECTURES:
        return None, [], None, f"Unknown model selection: {model_label}"

    architecture = ARCHITECTURES[model_label]
    try:
        display_image = _to_display_image(image)
        model, class_names, _img_size, preprocess = load_model(architecture)
        input_tensor = preprocess(display_image).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        predicted_index = int(np.argmax(probabilities))
        friendly_names = [_friendly_class_name(name) for name in class_names]
        label_scores = {
            friendly_names[i]: float(probabilities[i]) for i in range(len(class_names))
        }
        probability_rows = [
            [friendly_names[i], round(float(probabilities[i]), 6)]
            for i in np.argsort(probabilities)[::-1]
        ]

        cam_image = None
        cam_message = "Grad-CAM unavailable."
        try:
            cam_image = _generate_gradcam(
                model, architecture, input_tensor, predicted_index, display_image
            )
            cam_message = (
                "Grad-CAM shows image regions that influenced this model's score. "
                "It is exploratory and does not prove medical correctness."
            )
        except Exception as cam_exc:
            warnings.warn(f"Could not generate Grad-CAM: {cam_exc}")
            cam_message = f"Prediction succeeded, but Grad-CAM could not be generated: {cam_exc}"

        predicted_name = friendly_names[predicted_index]
        confidence = float(probabilities[predicted_index])
        status = (
            f"**Selected model:** {model_label}  \n"
            f"**Predicted dataset label:** {predicted_name}  \n"
            f"**Model score:** {confidence:.1%}  \n\n"
            f"{cam_message}  \n\n"
            "**Research/education only:** this model is not clinically validated and must not "
            "be used to diagnose, rule out, or guide treatment for a medical condition. "
            "Scores are not calibrated clinical probabilities."
        )
        return label_scores, probability_rows, cam_image, status
    except (FileNotFoundError, KeyError, ValueError, TypeError, RuntimeError, OSError,
            UnidentifiedImageError) as exc:
        return None, [], None, f"**Could not run prediction:** {exc}"
    except Exception as exc:  # Keep runtime errors visible in the UI without a traceback dump.
        warnings.warn(f"Unexpected inference error: {exc!r}")
        return None, [], None, f"**Unexpected inference error:** {type(exc).__name__}: {exc}"


def load_comparison_table() -> pd.DataFrame:
    path = RESULTS_DIR / "model_comparison.csv"
    if not path.is_file():
        return pd.DataFrame(columns=["model", "evaluation", "accuracy", "f1_macro", "recall_macro"])
    try:
        frame = pd.read_csv(path)
        if frame.empty:
            return pd.DataFrame(columns=frame.columns.tolist() or ["No results yet"])
        return frame
    except Exception as exc:
        return pd.DataFrame({"Comparison table could not be read": [str(exc)]})


COMPARISON_GUIDANCE = (
    "This table is populated by notebooks 02–05 after model evaluation. Run the relevant "
    "notebooks first, then select **Refresh table**. CLIP rows are zero-shot results; compare "
    "them with supervised models carefully."
)


def refresh_comparison():
    return load_comparison_table()


with gr.Blocks(title="Brain MRI Classification — Research Demo") as demo:
    gr.Markdown(
        "# Brain MRI Classification & Explainability\n"
        "Compare your trained Custom CNN, ResNet-18, and ViT-B/16 checkpoints with an "
        "interactive inference demo."
    )
    gr.Markdown(
        "> **Research/education only.** This is not a medical diagnostic tool. Model scores "
        "may be wrong, uncalibrated, or affected by dataset bias."
    )

    with gr.Tab("Predict an image"):
        with gr.Row():
            with gr.Column(scale=1):
                model_selector = gr.Dropdown(
                    choices=list(ARCHITECTURES.keys()),
                    value="ResNet-18",
                    label="Trained model checkpoint",
                    info="The corresponding models/*.pt file must exist. Train its notebook first.",
                )
                image_input = gr.Image(type="pil", label="Upload brain MRI image", height=320)
                predict_button = gr.Button("Classify image", variant="primary")
            with gr.Column(scale=1):
                prediction_output = gr.Label(label="Class scores", num_top_classes=4)
                probability_table = gr.Dataframe(
                    headers=["Class", "Softmax score"],
                    datatype=["str", "number"],
                    label="Scores by class (descending)",
                    interactive=False,
                )
        with gr.Row():
            cam_output = gr.Image(type="pil", label="Grad-CAM overlay", height=360)
            status_output = gr.Markdown("Upload an MRI image and select a trained model.")

        predict_button.click(
            fn=predict,
            inputs=[image_input, model_selector],
            outputs=[prediction_output, probability_table, cam_output, status_output],
        )

    with gr.Tab("Model comparison"):
        gr.Markdown("## Results from the training notebooks\n" + COMPARISON_GUIDANCE)
        comparison_table = gr.Dataframe(
            value=load_comparison_table(),
            label="Saved benchmark metrics",
            interactive=False,
            wrap=True,
        )
        gr.Button("Refresh table").click(fn=refresh_comparison, inputs=[], outputs=[comparison_table])

    with gr.Tab("Run instructions"):
        gr.Markdown(
            "## Before launching\n"
            "1. Install dependencies from `requirements.txt`.\n"
            "2. Download and prepare the dataset using notebook 01.\n"
            "3. Run notebook 02 to create `models/custom_cnn.pt`; run notebook 03 for "
            "`models/resnet18.pt`; run notebook 04 for `models/vit_b16.pt`.\n"
            "4. Run `python app.py` from the project root.\n\n"
            "The app does not train models. If a selected checkpoint is missing, choose a model "
            "whose checkpoint exists or run its training notebook."
        )


if __name__ == "__main__":
    demo.launch()
