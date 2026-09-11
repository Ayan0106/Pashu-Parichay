import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import argparse
import sys

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
# Model Loading
# =============================

def load_model(model_path: str):
    """Load a saved ConvNeXt-Tiny model checkpoint."""
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
# Single Image Prediction
# =============================

def predict(model, class_names, img_path: str, top_k: int = 3):
    """
    Predict the breed for a single image.

    Returns:
        pred_class (str)  : Top-1 predicted class name
        confidence (float): Confidence percentage
        top_k_preds (list): List of (class_name, prob%) tuples for top-k classes
    """
    image = Image.open(img_path).convert("RGB")
    input_tensor = val_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]

    top_probs, top_idxs = torch.topk(probs, k=min(top_k, len(class_names)))

    pred_class = class_names[top_idxs[0].item()]
    confidence = top_probs[0].item() * 100

    top_k_preds = [
        (class_names[idx.item()], round(prob.item() * 100, 2))
        for idx, prob in zip(top_idxs, top_probs)
    ]

    return pred_class, confidence, top_k_preds


# =============================
# CLI Entry Point
# =============================

def parse_args():
    parser = argparse.ArgumentParser(
        description="PashuParichay — Indian Bovine Breed Predictor"
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
        "--top-k", type=int, default=3,
        help="Number of top predictions to display (default: 3)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Loading model from: {args.model}")
    model, class_names = load_model(args.model)
    print(f"Model loaded. Classes: {class_names}\n")

    print(f"Predicting for: {args.image}")
    pred_class, confidence, top_k_preds = predict(
        model, class_names, args.image, top_k=args.top_k
    )

    print(f"\n{'='*40}")
    print(f"  Predicted Breed : {pred_class}")
    print(f"  Confidence      : {confidence:.2f}%")
    print(f"{'='*40}")

    if args.top_k > 1:
        print(f"\nTop-{args.top_k} Predictions:")
        for rank, (cls, prob) in enumerate(top_k_preds, 1):
            print(f"  {rank}. {cls:<20} {prob:.2f}%")
