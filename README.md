# 🫁 Pneumonia Detection from Chest X-Ray Images using CNN

An end-to-end Deep Learning project that uses a **Convolutional Neural Network (CNN)** built with TensorFlow and Keras to classify chest X-ray images into two categories: **NORMAL** and **PNEUMONIA**.

The project covers dataset exploration, image preprocessing, model development, training, evaluation, visualization, and prediction on new images.

> ⚠️ **Disclaimer:** This project is developed for educational and research purposes only. The model is experimental and is not a medical device. It must never be used for medical diagnosis or clinical decisions.

---

## ✨ Project Highlights

- CNN architecture developed from scratch using TensorFlow/Keras.
- Binary image classification: NORMAL vs PNEUMONIA.
- Memory-efficient image processing using `tf.data`.
- Image resizing and preprocessing (224 × 224).
- Stratified train-validation splitting.
- Class imbalance handling using class weights.
- Data augmentation to improve model generalization.
- Training callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, and BackupAndRestore.
- Model checkpoint saving and reloading.
- Comprehensive evaluation using multiple classification metrics.
- Confusion matrix and performance visualizations.
- Reusable `predict_image()` function for new X-ray images.

## 📂 Project Structure

```text
Pneumonia-Detection/
│
├── Pneumonia_Detection_Colab.ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── chest_xray/
│       ├── train/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       │
│       └── test/
│           ├── NORMAL/
│           └── PNEUMONIA/
│
└── models/
    ├── best_pneumonia_cnn.keras
    └── training_history.json
```

**Note:** The `data/` and `models/` directories are excluded from GitHub to avoid uploading large files. They are generated or populated locally when needed.

## 📊 Dataset

**Chest X-Ray Images (Pneumonia)** by Paul Mooney.

- [Original Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- [Hugging Face Dataset Mirror](https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia)

The notebook downloads the dataset from a public Hugging Face mirror.

The dataset contains chest X-ray images from pediatric patients aged 1–5 years at the Guangzhou Women and Children's Medical Center.

It contains two classes:

| Class | Description |
|---|---|
| NORMAL | Chest X-ray images labelled as normal |
| PNEUMONIA | Chest X-ray images labelled as pneumonia |

### Dataset Distribution

| Split | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Original Train | 1,341 | 3,875 | 5,216 |
| Original Test | 234 | 390 | 624 |

The original training split is divided into:

- **Training set:** 85%
- **Validation set:** 15%

The original test set remains separate and is used for final model evaluation.

The notebook calculates the actual image counts automatically.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow / Keras | Deep Learning and CNN development |
| NumPy | Numerical operations |
| Pandas | Data manipulation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Data splitting and evaluation metrics |
| Pillow | Image processing |
| Hugging Face Datasets | Dataset downloading |
| tqdm | Progress monitoring |
| Jupyter Notebook | Development environment |

## 🧠 CNN Architecture

The model is built from scratch using four convolutional blocks.

Each block includes:

- `Conv2D`
- `BatchNormalization`
- `ReLU`
- `MaxPooling2D`

The architecture uses 32, 64, 128, and 256 convolutional filters.

The final layers include:

- Global Average Pooling
- Dense layer with 128 neurons
- Dropout for regularization
- Sigmoid output layer

### Model Configuration

| Parameter | Value |
|---|---|
| Input image size | 224 × 224 × 3 |
| Batch size | 32 |
| Initial learning rate | 0.001 |
| Maximum epochs | 20 |
| Optimizer | Adam |
| Loss function | Binary Crossentropy |
| Output activation | Sigmoid |

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/HananeBouzidi06/Pneumonia-Detection.git

cd Pneumonia-Detection
```

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv

.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv

source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

**Note:** TensorFlow compatibility depends on the operating system, processor architecture, and Python version. Make sure to use a compatible Python environment. For Mac Intel, verify TensorFlow installation compatibility before running the notebook.

### 4. Run the Notebook

The notebook can be executed locally using **Visual Studio Code** or **Jupyter Notebook**.

1. Open `Pneumonia_Detection_Colab.ipynb`.
2. Select the Python environment where the dependencies were installed.
3. Run the notebook cells from top to bottom.
4. The dataset is downloaded automatically from Hugging Face when needed.
5. The model is trained and evaluated.

**Alternative:** Google Colab can also be used as an execution environment.

## 💾 Model Saving

The best model is saved in Keras format:

```text
models/best_pneumonia_cnn.keras
```

The training history is also saved:

```text
models/training_history.json
```

The notebook can reload an existing model to avoid unnecessary retraining.

To load the saved model:

```python
from tensorflow import keras

model = keras.models.load_model(
    "models/best_pneumonia_cnn.keras"
)
```

## 📈 Model Evaluation

The model is evaluated using several metrics:

| Metric | Description |
|---|---|
| Accuracy | Overall classification accuracy |
| Precision | Proportion of predicted pneumonia cases that are correct |
| Recall | Proportion of actual pneumonia cases correctly identified |
| F1-score | Harmonic mean of precision and recall |
| ROC AUC | Ability to distinguish between the two classes |
| Confusion Matrix | Distribution of correct and incorrect predictions |

The notebook also generates a classification report and training performance visualizations.

**Note:** Evaluation results are calculated during execution and depend on the training run. No fixed performance values are assumed.

## 🔍 Prediction on New Images

The project includes a reusable prediction function:

```python
label, confidence = predict_image(
    "path/to/chest_xray.jpg"
)

print("Prediction:", label)
print("Confidence:", confidence)
```

The function processes an input X-ray image and returns one of two predicted classes:

- `NORMAL`
- `PNEUMONIA`

The prediction score represents the model's confidence according to its output, not a medically validated probability.

## ⚠️ Limitations

- The dataset comes from a single medical center and contains pediatric images.
- The model may not generalize to adults or images from other hospitals.
- Bacterial and viral pneumonia are grouped into one class.
- The Hugging Face dataset mirror does not expose original patient identifiers.
- Validation splitting is stratified by class, not patient-level.
- Images from the same patient could potentially appear in different splits.
- The model has not undergone clinical validation.
- No explainability analysis or external validation has been performed.

## 🔭 Future Improvements

- Transfer learning using ResNet, EfficientNet, or DenseNet.
- Grad-CAM for model explainability.
- Cross-validation and external dataset evaluation.
- Decision threshold optimization and probability calibration.
- Extension to multiple pneumonia categories.

## 📚 References

- Kermany, D. S., et al. (2018). *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*. Cell.
- [Original Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- [Hugging Face Dataset Mirror](https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia)
- [TensorFlow Documentation](https://www.tensorflow.org/)

## 📝 License

The source code may be distributed under an appropriate open-source license.

The dataset is subject to its original licensing and usage terms. Please check the dataset provider's conditions before redistribution.

---

**Developed as an educational Deep Learning project using Python, TensorFlow, and Convolutional Neural Networks.**