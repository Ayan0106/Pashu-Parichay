# 🐄 Pashu Parichay

### Image-Based Indian Cattle & Buffalo Breed Recognition

**Pashu Parichay** is an AI-powered computer vision system designed to recognize Indian cattle and buffalo breeds from photographs. The project uses deep learning-based image classification to predict the breed of an animal from an input image and provides **Grad-CAM visual explanations** to highlight the regions that influenced the model's prediction.

The project is designed as a practical computer vision solution for applications such as livestock identification, breed awareness, agricultural technology, and digital livestock management.

---

## ✨ Key Features

* 🐄 **Breed Recognition from Images**
* 🧠 Deep learning-based image classification using **ConvNeXt-Tiny**
* 🔄 Data augmentation for improved generalization
* 📊 Training and validation performance monitoring
* 🧩 Confusion matrix analysis
* 🔥 **Grad-CAM explainability**
* 🎯 Confidence score for predictions
* 💾 Exportable PyTorch model checkpoint
* ⚡ GPU-accelerated training support

---

## 🎯 Problem Statement

Identifying livestock breeds manually can require significant domain knowledge and may become difficult when dealing with visually similar breeds.

Pashu Parichay aims to provide an automated computer vision approach where a user can provide an image of a cattle or buffalo and receive a predicted breed along with a confidence score.

The system can potentially support:

* Farmers and livestock owners
* Veterinary and agricultural applications
* Livestock documentation
* Breed awareness and education
* Digital livestock management systems
* AI-assisted agricultural platforms

---

## 🧠 System Overview

The current machine learning pipeline follows the workflow:

```text
                 Input Animal Image
                         │
                         ▼
                Image Preprocessing
                         │
                         ▼
                 Data Augmentation
                         │
                         ▼
                ConvNeXt-Tiny Model
                         │
                         ▼
                 Feature Extraction
                         │
                         ▼
                  Classification
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        Predicted Breed       Confidence Score
                                    │
                                    ▼
                              Grad-CAM
                                    │
                                    ▼
                       Visual Explanation Heatmap
```

---

## 🏗️ Model Architecture

The current implementation uses **ConvNeXt-Tiny** with pretrained weights.

The original classification head is replaced with a new linear layer corresponding to the number of livestock breed classes present in the dataset.

### Model Configuration

| Component              | Configuration                  |
| ---------------------- | ------------------------------ |
| Architecture           | ConvNeXt-Tiny                  |
| Pretrained Weights     | Torchvision pretrained weights |
| Input Size             | 224 × 224                      |
| Optimizer              | AdamW                          |
| Learning Rate          | 1e-4                           |
| Weight Decay           | 0.05                           |
| Loss                   | Cross Entropy Loss             |
| Label Smoothing        | 0.1                            |
| LR Scheduler           | Cosine Annealing               |
| Epochs                 | 30                             |
| Batch Size             | 32                             |
| Train/Validation Split | 80/20                          |
| Random Seed            | 42                             |
| Hardware               | CUDA GPU when available        |

---

## 🔄 Data Preprocessing

### Training Transformations

Training images are processed using:

* Resize to `224 × 224`
* Random horizontal flipping
* Random rotation up to 20°
* Color jitter
* Tensor conversion
* ImageNet normalization

### Validation Transformations

Validation images use:

* Resize to `224 × 224`
* Tensor conversion
* ImageNet normalization

The dataset is loaded using PyTorch's `ImageFolder`, where each breed is represented as a separate directory/class.

---

## 📊 Training Results

The current training run was performed for **30 epochs**.

### Best Validation Performance

* **Best validation accuracy:** ~**66.95%**
* **Best validation epoch:** **Epoch 22**
* Final training accuracy: ~**96.94%**

The validation accuracy is reported as validation performance rather than test performance because the current pipeline does not use a separate held-out test dataset.

### Training Progress

The model showed rapid improvement during the initial epochs before validation performance began to plateau.

```text
Epoch 1   → Validation Accuracy: 49.92%
Epoch 2   → Validation Accuracy: 58.01%
Epoch 3   → Validation Accuracy: 61.72%
Epoch 4   → Validation Accuracy: 66.10%
...
Epoch 22  → Validation Accuracy: 66.95%  ← Best
...
Epoch 30  → Validation Accuracy: 66.78%
```

The difference between training and validation accuracy indicates that the current model has some degree of **overfitting**, which is an important area for future improvement.

---

## 🔥 Explainability with Grad-CAM

Pashu Parichay incorporates **Gradient-weighted Class Activation Mapping (Grad-CAM)** to make model predictions more interpretable.

Instead of only returning:

```text
Predicted Breed: Gir
Confidence: 87.4%
```

the system can also generate a heatmap showing the image regions that contributed to the prediction.

### Explainability Pipeline

```text
Input Image
     │
     ▼
ConvNeXt-Tiny
     │
     ▼
Predicted Class
     │
     ▼
Gradients + Feature Activations
     │
     ▼
Grad-CAM Heatmap
     │
     ▼
Original Image + Heatmap Overlay
```

This provides a visual indication of where the model is focusing when making its prediction.

---

## 🖼️ Example Prediction

The inference pipeline supports predictions such as:

```text
Predicted Breed : Gir
Confidence      : XX.XX%
```

along with a corresponding Grad-CAM visualization.

> Add your actual prediction screenshots and Grad-CAM outputs to the `results/` or `assets/` directory before publishing the repository.

---

## 📁 Project Structure

Recommended repository structure:

```text
Pashu-Parichay/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── pashuparichay.ipynb
│
├── src/
│   ├── train.py
│   ├── predict.py
│   └── gradcam.py
│
├── models/
│   └── README.md
│
├── results/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── gradcam_examples/
│
├── assets/
│   ├── architecture.png
│   └── demo.png
│
└── app/
    └── application source code
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Pashu-Parichay.git
cd Pashu-Parichay
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Download the Indian bovine breed dataset used during development and arrange it using the following structure:

```text
data/
└── Indian_bovine_breeds/
    ├── Breed_1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── Breed_2/
    │   └── ...
    └── ...
```

The project uses PyTorch `ImageFolder`, so each directory represents one classification category.

> The dataset itself is not included in this repository.

---

## 🏋️ Training

Run:

```bash
python src/train.py --data_dir data/Indian_bovine_breeds
```

The training pipeline:

1. Loads the dataset
2. Creates an 80/20 train-validation split
3. Applies image preprocessing and augmentation
4. Initializes ConvNeXt-Tiny
5. Replaces the classification head
6. Trains using AdamW
7. Applies cosine learning-rate scheduling
8. Tracks training/validation loss and accuracy
9. Restores the best validation model
10. Saves the trained model checkpoint

---

## 🔮 Prediction

After training, an image can be passed to the inference pipeline:

```bash
python src/predict.py --image path/to/image.jpg
```

Example output:

```text
Predicted Breed : Gir
Confidence      : 87.42%
```

---

## 💾 Model Checkpoint

The trained model is exported as a PyTorch checkpoint containing:

```python
{
    "model_state": model.state_dict(),
    "class_names": class_names
}
```

This allows the model weights and corresponding class mapping to be restored for inference.

Large model files should preferably be distributed through GitHub Releases, Hugging Face Hub, or another suitable model-storage service rather than committed directly to the repository.

---

## 🧪 Evaluation

The project currently evaluates the model using:

* Training accuracy
* Validation accuracy
* Training loss
* Validation loss
* Confusion matrix
* Grad-CAM visual explanations

The confusion matrix helps identify breeds that are frequently confused with each other and can guide future dataset and model improvements.

---

## ⚠️ Current Limitations

The current implementation has several limitations:

1. The evaluation uses a train/validation split rather than a completely independent test set.
2. The model shows a noticeable training-validation accuracy gap, suggesting overfitting.
3. Breed recognition performance may decrease for images with poor lighting, unusual viewpoints, occlusion, or backgrounds different from the training data.
4. Visually similar breeds can be difficult to distinguish.
5. The current notebook is Kaggle-oriented and requires path/configuration cleanup for fully reproducible local execution.
6. The dataset and trained weights are not distributed directly with the repository.
7. Model predictions should be treated as AI-assisted predictions rather than definitive veterinary or breed certification.

---

## 🔮 Future Improvements

Planned improvements include:

* [ ] Add a dedicated held-out test set
* [ ] Improve class balancing
* [ ] Perform stronger augmentation strategies
* [ ] Tune ConvNeXt hyperparameters
* [ ] Experiment with EfficientNet and other lightweight architectures
* [ ] Add top-k predictions
* [ ] Improve confidence calibration
* [ ] Add automated image-quality checks
* [ ] Add unknown/low-confidence detection
* [ ] Optimize the model for mobile/edge deployment
* [ ] Integrate the trained model into the Pashu Parichay application
* [ ] Build an API-based inference service
* [ ] Improve Grad-CAM visualization
* [ ] Expand the dataset with real-world livestock images

---

## 🛠️ Tech Stack

### Machine Learning

* Python
* PyTorch
* Torchvision
* ConvNeXt-Tiny
* OpenCV
* NumPy
* scikit-learn

### Visualization

* Matplotlib
* Seaborn

### Development

* Jupyter Notebook
* Kaggle GPU
* Git
* GitHub

---

## 🎓 Project Context

**Pashu Parichay** is a computer vision project focused on applying deep learning to livestock breed recognition.

The project explores how image classification and explainable AI can be combined to build practical agricultural technology solutions.

---

## ⭐ Acknowledgements

* PyTorch and Torchvision for the deep learning framework
* The creators/providers of the Indian bovine breed dataset used for training
* The open-source computer vision and machine learning community

---

## 📌 Disclaimer

Pashu Parichay is an AI-based breed recognition system developed for educational and research purposes. Predictions may be incorrect, particularly for visually similar breeds or images outside the training distribution. The system should not be considered a substitute for professional veterinary, livestock, or breed-certification expertise.
