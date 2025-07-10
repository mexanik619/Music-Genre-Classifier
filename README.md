
# 🎵 Music Genre Classification with CNNs

This repository implements a music genre classification system using deep learning (Convolutional Neural Networks) on audio spectrograms. It explores multiple model versions with enhancements in architecture, preprocessing, and evaluation.

---

## 📂 Repository Structure

```
├── Colab_version.ipynb           # Final/Colab-optimized version
├── improvedmusicgenre.ipynb      # Enhanced version with augmentations and callbacks
├── musicgenre.ipynb              # Base version with basic CNN pipeline
└── README.md                     # Project documentation
```

---

## 🧠 Objective

The goal is to classify audio samples into one of **10 music genres** (Blues, Classical, Country, Disco, HipHop, Jazz, Metal, Pop, Reggae, Rock) using **mel spectrograms** as input features for CNN-based models.

---

## 📦 Dependencies

Make sure to install the following libraries:

```bash
pip install numpy pandas matplotlib seaborn librosa scikit-learn tensorflow
```

---

## 🎼 Dataset

Dataset used: [GTZAN Genre Collection](http://marsyas.info/downloads/datasets.html)

- Format: `.wav` files, 22050 Hz, 30 seconds each.
- Location: Place the dataset inside `C:/Datasets/Data/genres_original/` or change the path in the notebooks accordingly.

---

## 🧰 Feature Extraction

All audio files are converted to **Mel Spectrograms** using `librosa`:

```python
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
S_DB = librosa.power_to_db(S, ref=np.max)
```

- Shape standardized to `(128, 660)` for CNN input.
- Skips any problematic or corrupted files with error handling.

---

## 🏗️ Model Architectures

### 1. **Basic CNN** (`musicgenre.ipynb`)
- Simple 3-layer CNN
- Uses mel spectrograms directly
- Accuracy ~75%

```python
Conv2D → MaxPool → Dropout → Flatten → Dense
```

---

### 2. **Improved CNN** (`improvedmusicgenre.ipynb`)
- 4 Convolutional blocks with BatchNorm and Dropout
- Data augmentation using noise and pitch shifts
- Regularization (L2) on dense layers
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

```python
Conv2D → BN → Conv2D → MaxPool → Dropout (×4)
→ Flatten → Dense (L2) → Dropout → Output
```

---

### 3. **Colab Version** (`Colab_version.ipynb`)
- Final cleaned and modular version
- Wrapped in functions like `load_dataset_enhanced()`, `train_model_enhanced()`, etc.
- Training history and confusion matrix visualizations

---

## 🏃 Running the Project

1. **Extract Features and Load Data:**

```python
X, y, label_encoder = load_dataset_enhanced("C:/Datasets/Data/genres_original")
```

2. **Train/Test Split:**

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

3. **Train Model:**

```python
model, history = train_model_enhanced(X_train, y_train, X_test, y_test, label_encoder)
```

4. **Evaluate Model:**

```python
evaluate_model(model, X_test, y_test, label_encoder, history)
```

---

## 📊 Results

- **Accuracy:** Up to **92%** with the enhanced model.
- **Precision** and **Validation Accuracy** monitored during training.
- Confusion matrix and learning curves plotted using `matplotlib` and `seaborn`.

---

## 📈 Visualizations

- **Training vs Validation Accuracy**
- **Training vs Validation Loss**
- **Confusion Matrix**
- **Classification Report**

---

## 🧪 Enhancements Implemented

- 🧹 Dataset cleansing and error handling
- 📈 Stratified train-test splitting
- 📊 Visualizations and evaluation metrics
- 📦 Model checkpoints and adaptive learning
- 🔄 Augmented dataset for better generalization

---

## 🔮 Future Improvements

- Add real-time inference with microphone input
- Use pretrained models (e.g., VGGish or YAMNet)
- Implement genre probability ranking
- Deploy using Streamlit or Flask for interactive web app

---

## 📝 Acknowledgements

- [GTZAN Genre Dataset](http://marsyas.info/downloads/datasets.html)
- [LibROSA](https://librosa.org/)
- TensorFlow, Keras, NumPy, Seaborn, Matplotlib

---

## 📚 References

- [Music Genre Classification with CNN](https://towardsdatascience.com/music-genre-classification-with-cnns-cb71e41c5c31)
- [LibROSA Documentation](https://librosa.org/doc/main/index.html)
- [TensorFlow Keras API](https://www.tensorflow.org/api_docs/python/tf/keras)
