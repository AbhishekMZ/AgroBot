# AgroBot

Autonomous Weed Detection and Spraying Rover

AgroBot is an open-source robotics project designed to automate weed detection and precision spraying in agricultural fields. Leveraging computer vision and embedded systems, AgroBot aims to reduce manual labor and optimize chemical usage for sustainable farming.

---

## 🚜 Features
- **Real-time weed detection** using deep learning (object detection)
- **Precision spraying** via motorized chassis and sprayer
- **Raspberry Pi Zero 2 W** as main controller
- **Arduino** for low-level motor and sprayer control
- **Modular software** for easy extension and debugging

---

## 🛠️ Hardware Requirements
- Raspberry Pi Zero 2 W
- Camera module (compatible with Pi)
- Arduino (Uno/Nano)
- Motor driver (L298N or similar)
- Rover chassis with motors
- Solenoid valve or relay for sprayer
- Power supply (battery pack)

---

## 💻 Software Requirements
- Python 3.8+
- pip (Python package manager)
- (Optional) Virtual environment: `venv` or `conda`
- Arduino IDE (for uploading firmware)

---

## 📁 Project Structure
```
AgroBot/
├── app/                # Main application logic
├── communication/      # Serial and network communication modules
├── config/             # Configuration files and environment variables
├── core/               # Core robotics and control logic
├── data/               # Datasets and sample images
├── hal/                # Hardware Abstraction Layer (camera, chassis, sprayer)
├── training/           # Model training and quantization scripts
├── utils/              # Utility scripts
├── requirements-pi.txt # Pi-specific dependencies
├── requirements-dev.txt# Dev dependencies
├── main.py             # Entry point for AgroBot
├── sample_arduino.cpp  # Arduino firmware sample
├── structure.txt       # Project structure reference
├── .env                # Environment variables (do not commit sensitive info)
└── README.md           # This file
```

---

## ⚡ Quickstart

1. **Clone the repository:**
   ```sh
   git clone https://github.com/AbhishekMZ/AgroBot.git
   cd AgroBot
   ```
2. **(Recommended) Create a virtual environment:**
   ```sh
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On Linux/Mac
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements-dev.txt
   # For Raspberry Pi:
   pip install -r requirements-pi.txt
   ```
4. **Configure environment variables:**
   - Copy `.env.example` to `.env` and fill in required values.

---

## 🤖 Model Training & Deployment
- **Dataset:** Use annotated images with bounding boxes for weed/crop detection.
- **Model:** EfficientDet-Lite or YOLO (for object detection, optimized for TFLite)
- **Training:**
  - See `training/model_trainer.py` for training scripts.
  - See `training/model_quantizer.py` for quantization (TFLite conversion).
- **Deployment:**
  - Deploy the quantized `.tflite` model to the Raspberry Pi.
  - Inference is handled in `main.py` using the HAL camera interface.

---

## 🏃 Usage
- **Run the main application:**
  ```sh
  python main.py
  ```
- **Upload Arduino firmware:**
  - Use `sample_arduino.cpp` with the Arduino IDE.

---

## 🤝 Contributing
Pull requests and issues are welcome! Please follow best practices and open an issue to discuss major changes.

---

## 📚 References & Acknowledgements
- [TensorFlow Lite Models](https://www.tensorflow.org/lite/models)
- [YOLO Object Detection](https://pjreddie.com/darknet/yolo/)
- [Raspberry Pi Documentation](https://www.raspberrypi.org/documentation/)
- [Arduino Documentation](https://www.arduino.cc/en/Guide)

---

## ⚠️ License
Specify your license here (e.g., MIT, Apache 2.0, etc.)

---

For more details, see the `Understanding/` folder (not tracked by git).
