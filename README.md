#  MobileNet Image Classification & Sign Language Digits

> A two-part deep learning project: classify any image against **1,000 ImageNet categories** using MobileNet out-of-the-box, then fine-tune it to recognize **hand gestures for digits 0–9** from the Sign Language Digits Dataset.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-MobileNet-red?style=flat-square&logo=keras)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

---

## 📌 Overview

This project has two parts:

1. **ImageNet Prediction Demo** — Run the pretrained MobileNet on 6 custom sample images and decode the top predictions from 1,000 ImageNet classes. Zero training needed.
2. **Sign Language Digits Classifier** — Fine-tune MobileNet on the Sign Language Digits Dataset to classify hand gestures for digits **0 through 9** (10 classes).

---

## 🖼️ Sample Images — ImageNet Prediction Demo

These 6 images are passed through the pretrained MobileNet and classified in real time:

| | | |
|:---:|:---:|:---:|
| ![1](https://raw.githubusercontent.com/trisha2103/Mobile-Net-Image-Classification/main/1.jpg) | ![2](https://raw.githubusercontent.com/trisha2103/Mobile-Net-Image-Classification/main/2.jpg) | ![3](https://raw.githubusercontent.com/trisha2103/Mobile-Net-Image-Classification/main/3.jpg) |
| Sample 1 | Sample 2 | Sample 3 |
| ![4](https://raw.githubusercontent.com/trisha2103/Mobile-Net-Image-Classification/main/4.jpg) | ![5](https://raw.githubusercontent.com/trisha2103/Mobile-Net-Image-Classification/main/5.jpg) | ![6](https://raw.githubusercontent.com/trisha2103/Mobile-Net-Image-Classification/main/6.jpg) |
| Sample 4 | Sample 5 | Sample 6 |

> Each image is resized to **224×224**, preprocessed, and fed into MobileNet — which returns the top-5 predictions with confidence scores using `imagenet_utils.decode_predictions()`.

---

## 🚀 Features

- ✅ Out-of-the-box **ImageNet prediction** on custom images (1,000 classes)
- ✅ **Partial fine-tuning** — last 22 layers unfrozen for domain adaptation
- ✅ Custom output head for **10-class sign language classification**
- ✅ Automated dataset splitting into `train / valid / test`
- ✅ **10×10 confusion matrix** for full digit-by-digit evaluation
- ✅ MobileNet-specific preprocessing pipeline

---

## 🗂️ Dataset Structure — Sign Language Fine-Tuning

```
Sign-Language-Digits-Dataset/
├── train/
│   ├── 0/     └── 1/     └── 2/  ...  └── 9/    # ~170+ images each
├── valid/
│   ├── 0/     └── 1/     ...  └── 9/             # 30 images each
└── test/
    ├── 0/     └── 1/     ...  └── 9/             # 5 images each
```

> Dataset source: [Sign Language Digits Dataset](https://github.com/ardamavi/Sign-Language-Digits-Dataset)

---

## 🧠 Model Architecture — Fine-Tuned MobileNet

| Layer | Details |
|---|---|
| MobileNet (early layers) | **Frozen** — preserves ImageNet features |
| MobileNet (last 22 layers) | **Trainable** — adapts to sign language gestures |
| Reshape | `(1024,)` — from `mobile.layers[-5].output` |
| Dense (output) | 10 units · Softmax — digits 0–9 |

> Unlike VGG16/VGG19 where all base layers are frozen, this fine-tune **unfreezes the last 22 MobileNet layers** for deeper domain adaptation while keeping training lightweight.

---

## 📖 How It Works

### Part 1 — ImageNet Prediction

```python
mobile = tf.keras.applications.mobilenet.MobileNet()

def prepare_image(file):
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

preprocessed_image = prepare_image('1.jpg')
predictions = mobile.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)
```

### Part 2 — Sign Language Fine-Tuning

#### Step 1 — Organize the Dataset

```python
for i in range(0, 10):
    shutil.move(f'{i}', 'train')
    valid_samples = random.sample(os.listdir(f'train/{i}'), 30)
    for j in valid_samples:
        shutil.move(f'train/{i}/{j}', f'valid/{i}')
    test_samples = random.sample(os.listdir(f'train/{i}'), 5)
    for k in test_samples:
        shutil.move(f'train/{i}/{k}', f'test/{i}')
```

#### Step 2 — Build Data Generators

```python
train_batches = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input
).flow_from_directory(train_path, target_size=(224, 224), batch_size=10)
```

#### Step 3 — Build the Fine-Tuned Model

```python
x = mobile.layers[-5].output
x = tf.keras.layers.Reshape(target_shape=(1024,))(x)
output = Dense(units=10, activation='softmax')(x)
model = Model(inputs=mobile.input, outputs=output)

for layer in model.layers[:-22]:
    layer.trainable = False
```

#### Step 4 — Train & Evaluate

```python
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_batches, validation_data=valid_batches, epochs=10, verbose=2)

# Confusion matrix across all 10 digit classes
cm_plot_labels = ['0','1','2','3','4','5','6','7','8','9']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
```

---

## 📊 Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss | Categorical Crossentropy |
| Batch Size | 10 |
| Epochs | 10 |
| Image Size | 224 × 224 |
| Trainable layers | Last 22 of MobileNet + Dense head |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| 🧠 `TensorFlow / Keras` | Model building & training |
| 📱 `MobileNet` | Pretrained ImageNet model |
| 🔢 `imagenet_utils` | Decode top-5 ImageNet predictions |
| 📊 `scikit-learn` | 10×10 confusion matrix |
| 📈 `matplotlib` | Visualization |
| 🐍 `NumPy` | Numerical operations |

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/trisha2103/Mobile-Net-Image-Classification.git
cd Mobile-Net-Image-Classification

# 2. Install dependencies
pip install tensorflow scikit-learn matplotlib numpy

# 3. For the ImageNet demo — sample images 1.jpg–6.jpg are already in the repo

# 4. For sign language fine-tuning — download the dataset:
#    https://github.com/ardamavi/Sign-Language-Digits-Dataset
#    Place digit folders (0–9) in Sign-Language-Digits-Dataset/

# 5. Launch the notebook
jupyter notebook "MobileNet - Image Classification.ipynb"
```

> 💡 GPU recommended for faster training. TensorFlow falls back to CPU automatically.

---

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👩‍💻 Author

**Trisha** — [@trisha2103](https://github.com/trisha2103)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">Made with ❤️ using TensorFlow & MobileNet</p>
