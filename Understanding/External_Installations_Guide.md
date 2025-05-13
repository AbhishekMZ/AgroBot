# AgroBot: External Installations and Setup Guide

This document outlines software and tools required for the AgroBot project that need to be downloaded or installed separately from the Python package managers (`pip`).

## I. Raspberry Pi Zero 2 W Setup

### 1. Raspberry Pi OS (Operating System)

*   **Requirement:** An ARM-based Raspberry Pi OS image.
*   **Crucial Note:** Do NOT use `rpd_x86` or `i386` ISO images for the Pi Zero 2 W hardware. These are for running Raspberry Pi Desktop on x86 PCs.
*   **Recommended Method:** Use the **Raspberry Pi Imager** tool.
    *   Download from: [https://www.raspberrypi.com/software/](https://www.raspberrypi.com/software/)
    *   Choose an appropriate OS for the Pi Zero 2 W (e.g., "Raspberry Pi OS (32-bit)" based on Bullseye or Bookworm).
*   **Alternative (Direct Download):**
    *   Visit: [https://www.raspberrypi.com/software/operating-systems/](https://www.raspberrypi.com/software/operating-systems/)
    *   Download the correct `.img` file (or a `.zip` containing it) for your Pi model.

### 2. TensorFlow Lite Runtime (`tflite-runtime`)

*   **Requirement:** A specific Python wheel (`.whl`) file compatible with your Raspberry Pi's architecture (e.g., `armv7l` for 32-bit OS, `aarch64` for 64-bit OS) and its Python 3 version.
*   **Installation Steps on Pi:**
    1.  After installing Raspberry Pi OS, open a terminal and check your Python version: `python3 --version`.
    2.  Go to the **official TensorFlow Lite Python quickstart guide**: [https://www.tensorflow.org/lite/guide/python](https://www.tensorflow.org/lite/guide/python)
    3.  Find and download the `.whl` file that matches your Pi's Python version and architecture.
    4.  Copy the downloaded `.whl` file to your Raspberry Pi.
    5.  Install it using pip: `pip3 install name_of_the_tflite_runtime_wheel.whl`.
*   **Example (Illustrative - always check official source):**
    *   `tflite_runtime-2.14.0-cp311-cp311-linux_armv7l.whl` (This was an example you found, ensure it matches your Pi's Python 3.11 on a 32-bit ARMv7 OS).

### 3. System Libraries (via `apt` on Raspberry Pi)

Before installing Python packages via `pip`, it's often necessary to install system-level dependencies:

```bash
sudo apt update
sudo apt full-upgrade
sudo apt install python3-dev python3-pip libatlas-base-dev
# For OpenCV (if not sufficiently covered by opencv-python-headless dependencies):
sudo apt install libjpeg-dev libpng-dev libtiff-dev libavformat-dev libswscale-dev libgtk-3-dev
# For PiCamera2 (if using):
sudo apt install python3-picamera2
# For RPi.GPIO and gpiozero (often pre-installed or available via apt):
sudo apt install python3-rpi.gpio python3-gpiozero
```

## II. Development PC Setup (Windows/Linux/macOS)

### 1. Git (Version Control)

*   **Requirement:** Essential for managing source code.
*   **Download:** [https://git-scm.com/](https://git-scm.com/)

### 2. Python
*   **Requirement:** A recent version of Python 3 (e.g., 3.9, 3.10, 3.11).
*   **Download:** [https://www.python.org/downloads/](https://www.python.org/downloads/)

### 3. Integrated Development Environment (IDE)

*   **Recommendation:** VS Code, PyCharm, or your preferred Python IDE.
*   **Download (VS Code):** [https://code.visualstudio.com/](https://code.visualstudio.com/)

### 4. Arduino IDE

*   **Requirement:** If using an Arduino microcontroller for motor control or sensor input.
*   **Download:** [https://www.arduino.cc/en/software](https://www.arduino.cc/en/software)

### 5. Image Annotation Tool

*   **Requirement:** For labeling your image dataset (e.g., drawing bounding boxes for weeds/crops).
*   **Options:**
    *   **LabelImg:** Often installed via `pip install labelimg` or from GitHub releases. ([https://github.com/HumanSignal/labelImg](https://github.com/HumanSignal/labelImg))
    *   **CVAT (Computer Vision Annotation Tool):** Powerful, can be run via Docker or used as a web service. ([https://www.cvat.ai/](https://www.cvat.ai/))
    *   **VGG Image Annotator (VIA):** Lightweight, HTML-based. ([https://www.robots.ox.ac.uk/~vgg/software/via/](https://www.robots.ox.ac.uk/~vgg/software/via/))
    *   **Label Studio:** Open source data labeling tool. ([https://labelstud.io/](https://labelstud.io/))

### 6. (Optional) NVIDIA CUDA Toolkit & cuDNN

*   **Requirement:** If you have an NVIDIA GPU and want to accelerate TensorFlow model training on your PC.
*   **Download:** From the NVIDIA Developer website. Ensure versions are compatible with your TensorFlow version and NVIDIA driver.

## III. General Recommendations

*   **Virtual Environments:** Always use virtual environments (e.g., `venv`) for Python projects on both your PC and the Raspberry Pi to manage dependencies and avoid conflicts.
    *   PC: `python -m venv .venv`
    *   Pi: `python3 -m venv .venv-pi`
*   **`pip-compile` (from `pip-tools`):** Consider using `pip-tools` to manage your requirements. You create a `.in` file (e.g., `requirements-dev.in`) with your top-level dependencies, and `pip-compile` generates the fully pinned `requirements.txt` file.
    *   Install: `pip install pip-tools`
    *   Use: `pip-compile requirements-dev.in -o requirements-dev.txt`
