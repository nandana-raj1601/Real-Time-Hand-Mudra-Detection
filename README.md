# Hand Mudra Detection with TensorFlow Object Detection API

This project builds a **real-time hand mudra detection system** using the TensorFlow Object Detection API. It identifies 5 classical Indian mudras from webcam input using a custom-trained object detection model.

### 🎥 Demo
![Mudra Detection in Action](Screenshots/Demo.gif)

---

## 📌 Features

* 📷 Real-time hand gesture (mudra) detection using webcam
* 🧠 Custom-trained object detection model (SSD or EfficientDet)
* ✅ Trained to detect 5 mudras:

  * **Chandrakala**
  * **Kartarimukha**
  * **Katakamukha**
  * **Musti**
  * **Pataka**
* 📊 Evaluation with TensorBoard (loss, mAP)
* 💾 Exported and tested on unseen webcam data


---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/nandana-raj1601/Real-Time-Hand-Mudra-Detection.git
cd Real_Time-Hand-Mudra_Detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run real-time detection

Make sure your webcam is connected:

```bash
python detect_mudras_realtime.py
```



---

## 📊 Training Overview

| Metric           | Value              |
| ---------------- | ------------------ |
| Final Total Loss | \~0.15             |
| Model Used       | SSD / EfficientDet |
| Training Steps   | 10,000             |
| mAP\@0.5 (Eval)  | \~0.97             |

> ✅ Trained using TensorFlow 2.13.0 on Google Colab (CPU runtime)

---

## 🔬 How It Works

1. **Dataset**: Custom dataset of annotated mudra images in Pascal VOC format (XML)
2. **TFRecord Generation**: From XML files using preprocessing scripts
3. **Model Training**: Via `model_main_tf2.py` with your pipeline config
4. **Evaluation**: TensorBoard used to track training/eval metrics
5. **Export**: Model exported using `exporter_main_v2.py`
6. **Inference**: Real-time webcam detection using OpenCV + TensorFlow

---

## 🛠️ Tech Stack

* TensorFlow 2.13.0
* TensorFlow Object Detection API
* Python 3.9+
* OpenCV
* Google Colab

---

## 📄 License

MIT License © 2025

---

## 🙌 Acknowledgements

* [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
* OpenCV
* Indian classical mudra traditions from dance and yoga

