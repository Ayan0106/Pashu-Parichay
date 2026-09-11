import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

# =============================
# Configuration
# =============================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =============================
# Grad-CAM Engine
# =============================

class ConvNeXtGradCAM:
    """
    Gradient-weighted Class Activation Mapping for ConvNeXt models.

    Hooks into a target layer to capture activations and gradients,
    then produces a heatmap highlighting discriminative image regions.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None
        self.handlers = []

        self.handlers.append(
            target_layer.register_forward_hook(self._save_activation)
        )
        self.handlers.append(
            target_layer.register_full_backward_hook(self._save_gradient)
        )

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        """Remove registered hooks. Always call after use."""
        for h in self.handlers:
            h.remove()

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Generate a normalised Grad-CAM heatmap [0, 1].

        Args:
            input_tensor: Preprocessed image tensor (1, C, H, W).
            class_idx: Target class index. Uses argmax if None.

        Returns:
            heatmap: 2-D numpy array, values in [0, 1].
        """
        self.model.eval()

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        # Global average-pool gradients over spatial dims
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Weighted sum of activation maps
        cam = torch.sum(weights * self.activations, dim=1).squeeze()

        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam.cpu().numpy()


# =============================
# Model Loading
# =============================

def load_model(model_path: str):
    """Load a saved ConvNeXt-Tiny checkpoint."""
    checkpoint = torch.load(model_path, map_location=DEVICE)
    class_names = checkpoint['class_names']
    num_classes = len(class_names)

    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    model.load_state_dict(checkpoint['model_state'])
    model.to(DEVICE)
    model.eval()

    return model, class_names


# =============================
# Overlay Helper
# =============================

def overlay_heatmap(image_np: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.4) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap onto an RGB image.

    Args:
        image_np  : Original image as (H, W, 3) uint8 numpy array.
        heatmap   : Normalised [0,1] heatmap (any spatial size).
        alpha     : Blending weight for the heatmap (default 0.4).

    Returns:
        superimposed: Blended image as (H, W, 3) uint8 numpy array.
    """
    h, w = image_np.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized),
        cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    superimposed = np.uint8(image_np * (1 - alpha) + heatmap_rgb * alpha)
    return superimposed


# =============================
# Main Visualisation Function
# =============================

def visualize_gradcam(model, class_names, img_path: str,
                      class_idx: int = None, save_path: str = None):
    """
    Run Grad-CAM on a single image and display / save the result.

    Args:
        model       : Loaded ConvNeXt-Tiny model.
        class_names : List of class names.
        img_path    : Path to the input image.
        class_idx   : Target class index (None → argmax prediction).
        save_path   : If given, saves the figure to this path.
    """
    image = Image.open(img_path).convert("RGB")
    input_tensor = val_transform(image).unsqueeze(0).to(DEVICE)

    # Hook into the last ConvNeXt block
    target_layer = model.features[-1][-1]
    cam_engine = ConvNeXtGradCAM(model, target_layer)

    heatmap = cam_engine.generate_heatmap(input_tensor, class_idx=class_idx)
    cam_engine.remove_hooks()

    img_np = np.array(image)
    superimposed = overlay_heatmap(img_np, heatmap)

    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)

    pred_class = class_names[pred.item()]
    confidence = conf.item() * 100

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(image)
    axes[0].set_title(f"Prediction: {pred_class}\nConfidence: {confidence:.2f}%",
                      fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(superimposed)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=13)
    axes[1].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved Grad-CAM figure to: {save_path}")

    plt.show()

    print(f"\nPredicted Breed : {pred_class}")
    print(f"Confidence      : {confidence:.2f}%")


# =============================
# CLI Entry Point
# =============================

def parse_args():
    parser = argparse.ArgumentParser(
        description="PashuParichay — Grad-CAM Visualiser"
    )
    parser.add_argument(
        "--model", type=str, default="bovine_model.pth",
        help="Path to the saved model checkpoint (.pth)"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the input image"
    )
    parser.add_argument(
        "--class-idx", type=int, default=None,
        help="Target class index for Grad-CAM (default: argmax prediction)"
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Optional path to save the output figure (e.g. output.png)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Loading model from: {args.model}")
    model, class_names = load_model(args.model)
    print(f"Classes: {class_names}\n")

    visualize_gradcam(
        model, class_names,
        img_path=args.image,
        class_idx=args.class_idx,
        save_path=args.save
    )
