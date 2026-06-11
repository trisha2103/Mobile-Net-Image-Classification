# 📱 MobileNet Image Classification

> Classify any image against **1,000 ImageNet categories** using Google's pretrained **MobileNet** model — no training required.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-MobileNet-red?style=flat-square&logo=keras)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

---

## 📌 Overview

This project uses the **pretrained MobileNet** model to classify images straight out of the box — no fine-tuning, no dataset preparation. Load an image, preprocess it, and get back the **top-5 predicted labels** with confidence scores from ImageNet's 1,000 categories.

---

## 🖼️ Sample Images

These 6 images are included in the repo and classified by MobileNet:

| | | |
|:---:|:---:|:---:|
| ![1](https://raw.githubusercontent.com/trisha2103/Mobile-Net-Image-Classification/main/1.jpg) | ![2](https://raw.githubusercontent.com/trisha2103/Mobile-Net-Image-Classification/main/2.jpg) | ![3](https://raw.githubusercontent.com/trisha2103/Mobile-Net-Image-Classification/main/3.jpg) |
| Sample 1 | Sample 2 | Sample 3 |
| ![4](https://raw.githubusercontent.com/trisha2103/Mobile-Net-Image-Classification/main/4.jpg) | ![5](https://raw.githubusercontent.com/trisha2103/Mobile-Net-Image-Classification/main/5.jpg) | ![6](https://raw.githubusercontent.com/trisha2103/Mobile-Net-Image-Classification/main/6.jpg) |
| Sample 4 | Sample 5 | Sample 6 |

> Each image is resized to **224×224**, preprocessed with MobileNet's normalization, and decoded into the top-5 ImageNet class predictions with confidence scores.

---

## 🚀 Features

- ✅ Zero training — uses pretrained **ImageNet weights** directly
- ✅ Classifies any image into **1,000 categories**
- ✅ Returns **top-5 predictions** with confidence scores
- ✅ MobileNet-specific preprocessing built in
- ✅ Lightweight and fast — runs on CPU

---

## 📖 How It Works

### Step 1 — Load Pretrained MobileNet

```python
mobile = tf.keras.applications.mobilenet.MobileNet()
```

### Step 2 — Preprocess the Image

```python
def prepare_image(file):
    img_path = 'mobile_net_samples/'
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
```

### Step 3 — Predict & Decode

```python
preprocessed_image = prepare_image('1.jpg')
predictions = mobile.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)
print(results)
```

### Example Output

```
[[('n02504013', 'Indian_elephant', 0.7342),
  ('n01871265', 'tusker', 0.1923),
  ('n02504458', 'African_elephant', 0.0512),
  ...]]
```

---

## 🗂️ Project Structure

```
Mobile-Net-Image-Classification/
├── 1.jpg                                  # Sample images
├── 2.jpg
├── 3.jpg
├── 4.jpg
├── 5.jpg
├── 6.jpg
├── MobileNet - Image Classification.ipynb # Main notebook
├── LICENSE
└── README.md
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| 🧠 `TensorFlow / Keras` | Load & run MobileNet |
| 📱 `MobileNet` | Pretrained ImageNet model |
| 🔢 `imagenet_utils` | Decode top-5 predictions |
| 📈 `matplotlib` | Display images inline |
| 🐍 `NumPy` | Array operations |

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/trisha2103/Mobile-Net-Image-Classification.git
cd Mobile-Net-Image-Classification

# 2. Install dependencies
pip install tensorflow matplotlib numpy

# 3. Launch the notebook
jupyter notebook "MobileNet - Image Classification.ipynb"
```

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
