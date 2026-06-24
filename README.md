# Object Detection API (Faster R-CNN)

A production-ready REST API for object detection using a deep learning model based on Faster R-CNN.

---

## 🚀 Overview

This project demonstrates how to deploy a deep learning model as a scalable API.
It allows users to send an image and receive detected objects with their bounding boxes, labels, and confidence scores.

---

## 🧠 Use Cases

* Object detection in images
* Fruit recognition (custom dataset)
* Automated dataset annotation
* Computer vision prototyping

---

## Demo

![Demo](Sreenshot.png)

---

## 🏗️ Tech Stack

* Backend: FastAPI (Python)
* Machine Learning: PyTorch (CPU)
* Computer Vision: Torchvision (Faster R-CNN)
* Image Processing: Pillow
* Numerical Computing: NumPy
* Server: Uvicorn
* API: REST

---

## ⚙️ Installation

```bash
git clone https://github.com/Arsene666/Classifier-api.git
cd Classifier-api
pip install -r requirements.txt
```

---

## ▶️ Run the project

```bash
uvicorn main:app --reload
```

---

## 🔥 Example Request

```bash
POST /predict (multipart/form-data)
file: image.jpg
```

---

## 📊 Example Response

```json
{
  "labels": ["apple", "banana"],
  "scores": [0.98, 0.87],
  "boxes": [
    { "x1": 34.2, "y1": 12.1, "x2": 120.5, "y2": 98.3 },
    { "x1": 200.0, "y1": 50.0, "x2": 300.0, "y2": 180.0 }
  ]
}
```

---

## 📂 Project Structure

```
.
├── src/              # API and inference logic
├── models/           # trained model weights (.pth)
├── notebooks/        # training / experiments (to add)
├── tests/            # unit tests (to add)
```

---

## 🧪 Model Details

* Architecture: Faster R-CNN with ResNet-50 FPN backbone
* Framework: PyTorch
* Inference: CPU
* Classes:

  * apple
  * avocado
  * banana
  * guava
  * kiwi
  * mango
  * orange
  * peach
  * pineapple

---

## 🧪 Future Improvements

* Add training pipeline notebook
* Add evaluation metrics (mAP, precision, recall)
* Dockerize the application
* Deploy the API (Render / Railway / Hugging Face Spaces)
* Add batch inference

---

## 👤 Author

Arsène
