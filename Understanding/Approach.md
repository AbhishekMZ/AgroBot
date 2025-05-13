# AgroBot Project: Development Approach

This document outlines a phased approach to develop the AgroBot, an AI-powered robot for weed detection and elimination in agricultural fields, based on the project synopsis and technical restrictions.

## Phase 1: Project Setup and Component Acquisition

**Objective:** Gather all necessary hardware and set up the development environment.

**Tasks:**
1.  **Hardware Procurement:**
    *   **Processing Unit:** Raspberry Pi Zero 2 W.
    *   **Microcontroller:** Arduino (e.g., Arduino Uno/Nano) for motor and actuator control.
    *   **Camera:** Compatible camera module for Raspberry Pi (e.g., Pi Camera Module).
    *   **Rover Chassis:** Frame, wheels, motors (e.g., DC geared motors with drivers like L298N).
    *   **Spraying Mechanism:** Small pump, nozzle, relay module, tubing, and a container for herbicide/pesticide.
    *   **Power Supply:** Batteries for Pi, Arduino, motors, and pump.
    *   **Miscellaneous:** Jumper wires, SD card for Pi, USB cables.
2.  **Software Environment Setup (Raspberry Pi):**
    *   Install Raspberry Pi OS Lite.
    *   Install Python 3.
    *   Install necessary libraries:
        *   TensorFlow Lite runtime.
        *   OpenCV.
        *   Libraries for camera interaction (e.g., `picamera`).
        *   Libraries for serial communication (e.g., `pyserial` for Arduino communication).
3.  **Software Environment Setup (Development Machine):**
    *   IDE for Python development (e.g., VS Code).
    *   Arduino IDE for firmware development.
    *   Image annotation tool (e.g., LabelImg, CVAT).

## Phase 2: Data Collection and Preparation

**Objective:** Create a robust dataset for training the weed detection model.

**Tasks:**
1.  **Image Acquisition:**
    *   Collect a diverse dataset of images from crop fields.
    *   Ensure variability: different lighting conditions, weather, crop stages, weed types, and soil backgrounds.
    *   Use the selected camera module for consistency with the rover's setup.
2.  **Dataset Annotation:**
    *   Manually label images, clearly differentiating between "crop" and "weed."
    *   Use bounding boxes for object detection if that's the chosen approach (e.g., with SSD-MobileNet or EfficientDet-Lite).
3.  **Data Augmentation:**
    *   Apply augmentation techniques (rotation, scaling, flipping, brightness/contrast changes) to increase dataset size and model robustness.

## Phase 3: AI Model Development (Weed Detection)

**Objective:** Train and optimize a lightweight CNN model for accurate weed detection on the Raspberry Pi Zero 2 W.

**Tasks:**
1.  **Model Selection/Architecture Design:**
    *   Based on `Restrictions.txt`, consider:
        *   **TensorFlow Lite Models:** MobileNetV1/V2 (for classification if simpler) or EfficientDet-Lite / SSD-MobileNet (for object detection).
        *   **Custom CNN:** If necessary, design a small CNN architecture, keeping performance on Pi Zero 2 W in mind.
    *   Prioritize models that can be easily converted to TensorFlow Lite.
2.  **Model Training:**
    *   Use a framework like TensorFlow/Keras on a more powerful machine for training.
    *   Split the dataset into training, validation, and test sets.
    *   Consider using transfer learning from pre-trained models (e.g., MobileNet pre-trained on ImageNet) to speed up training and improve performance with limited data.
3.  **Model Optimization for Edge Deployment:**
    *   **Conversion to TensorFlow Lite (.tflite):** Convert the trained model.
    *   **Quantization:** Apply post-training quantization (INT8 recommended) to reduce model size and improve inference speed on the Pi Zero 2 W.
    *   **Performance Profiling:** Test inference speed and accuracy on the Pi Zero 2 W with sample images. Aim for input resolutions like 160x160 or 224x224.
4.  **Model Evaluation:**
    *   Evaluate using metrics like precision, recall, F1-score, and accuracy on the test set.
    *   Test specifically for false positives (spraying crops) and false negatives (missing weeds).

## Phase 4: Rover Hardware Assembly and Integration

**Objective:** Build the physical rover and integrate all electronic components.

**Tasks:**
1.  **Chassis Assembly:** Mount motors, wheels, and caster (if any) to the chassis.
2.  **Electronics Mounting:** Securely mount Raspberry Pi, Arduino, motor driver, camera, and power distribution system.
3.  **Wiring:**
    *   Connect camera to Raspberry Pi.
    *   Connect motor driver to Arduino and motors.
    *   Establish serial communication between Raspberry Pi (USB) and Arduino (USB/TX-RX pins).
    *   Connect the relay module for the sprayer to an Arduino digital pin.
    *   Connect the pump to the relay and power supply.
4.  **Spraying Mechanism Setup:** Assemble the container, pump, tubing, and nozzle. Ensure it's leak-proof and functional.
5.  **Power System:** Implement a power solution for all components, ensuring sufficient current and voltage.

## Phase 5: Software Development for Rover Control

**Objective:** Develop the software for both Raspberry Pi (main logic) and Arduino (low-level control).

**Tasks:**
1.  **Arduino Firmware Development (C/C++):**
    *   Write code to:
        *   Receive commands from Raspberry Pi via serial (e.g., "FORWARD", "STOP", "SPRAY", "RESUME").
        *   Control motors based on commands (e.g., drive forward, halt).
        *   Activate/deactivate the sprayer relay based on commands.
    *   Implement basic safety features if possible (e.g., stop if communication lost).
2.  **Raspberry Pi Main Control Script (Python):**
    *   **Camera Interface:** Capture images/video stream.
    *   **Image Preprocessing:** Resize, normalize images to match model input requirements.
    *   **Inference Engine:** Load the TFLite model and perform inference on captured frames.
    *   **Detection Logic:**
        *   Interpret model output (class probabilities or bounding box coordinates).
        *   Implement a threshold to classify as weed/crop.
        *   If multiple detections, decide on targeting.
    *   **Decision Making & Actuation Control:**
        *   If weed detected: Send "STOP" command to Arduino.
        *   After stopping: Send "SPRAY" command to Arduino for a short duration.
        *   After spraying: Send "RESUME" (or "FORWARD") command to Arduino.
        *   If no weed: Send "FORWARD" command (or maintain forward motion).
    *   **Serial Communication:** Implement robust serial communication with the Arduino.
    *   **Logging:** Log detections, actions, and potential errors for later analysis and optimization.

## Phase 6: System Testing, Deployment, and Iteration

**Objective:** Test the fully integrated system, deploy it in a representative environment, and iteratively improve its performance.

**Tasks:**
1.  **Modular Testing:** Test each component individually (camera, model inference on Pi, motor control via Arduino, sprayer).
2.  **Integrated System Testing (Controlled Environment):**
    *   Test rover movement, image capture, weed detection, and spraying sequence in a lab or small garden setting with known crop and weed samples.
    *   Debug communication issues between Pi and Arduino.
    *   Tune detection thresholds and sprayer duration.
3.  **Field Deployment and Testing:**
    *   Deploy the rover in an actual agricultural field.
    *   Monitor performance under real-world conditions (varying light, terrain, plant density).
    *   Collect data on detection accuracy, spray effectiveness, and rover traversal.
4.  **Feedback and Optimization:**
    *   Analyze logs and field test results.
    *   Retrain/fine-tune the AI model with newly collected field data if necessary.
    *   Adjust hardware (e.g., camera angle, nozzle position) or software parameters for better performance.
    *   Address any mechanical or electrical issues.

## Phase 7: Documentation and Future Enhancements (Optional)

**Objective:** Document the project and consider potential future improvements.

**Tasks:**
1.  **Documentation:**
    *   User manual for operation.
    *   Technical documentation for design, code, and hardware setup.
2.  **Potential Future Enhancements:**
    *   Autonomous navigation (e.g., using GPS, line following, or SLAM).
    *   Improved power efficiency (e.g., solar panels).
    *   Ability to distinguish between different types of weeds or crops.
    *   Mechanical weed removal options in addition to or instead of spraying.
    *   Cloud connectivity for remote monitoring and data collection.

This approach provides a structured path for the AgroBot project. Flexibility will be key, and iterations within each phase are expected as challenges arise.
