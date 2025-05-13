# I. Initial Analysis and Understanding

## A. Analyzing the Reference Codebase (`C:\Users\Mishael Abhishek\Projects\AgroBot\AgroBot_Reference`)

Your goal here is to understand its architecture, algorithms, data handling, and identify reusable components or establish a performance baseline.

1.  **Explore Directory Structure:** Familiarize yourself with how the code and data are organized within `AgroBot_Reference`.
2.  **Identify Key Components:**
    *   **Model Files:** Look for pre-trained model files (e.g., `.pb`, `.h5`, `.pth`, or already converted `.tflite`).
    *   **Training Scripts:** Scripts used for training or fine-tuning the model. Note the framework used (TensorFlow, Keras, PyTorch).
    *   **Inference Scripts:** Code demonstrating how to load the model and make predictions.
    *   **Data Preprocessing:** How is image data loaded, resized, normalized, or augmented?
    *   **Dataset:** Locate the image dataset. Understand its structure (folders for classes, annotation formats like XML, JSON, TXT).
3.  **Code Review (Key Scripts):**
    *   Understand the model architecture if defined in code.
    *   Note dependencies and libraries used.
    *   Analyze the workflow from input image to output prediction.
4.  **Questions to Answer:**
    *   What specific ML model/architecture is used? Does it align with `restrictions.txt`?
    *   What are its reported or observable accuracy and performance metrics (if any)?
    *   What is the format and size of the input data it expects?
    *   Can parts of this code be adapted for the Raspberry Pi Zero 2 W, or will it require significant modification/rewrite?

**B. Interpreting `restrictions.txt` and `Synopsis.txt`**

These are your guiding documents.

1.  **`restrictions.txt` - Key Takeaways:**
    *   **Target Hardware:** Raspberry Pi Zero 2 W. This dictates strict limits on computational power and memory.
    *   **ML Models:** TensorFlow Lite (TFLite) is heavily favored. Specific recommended lightweight architectures include MobileNetV1/V2, EfficientDet-Lite. OpenCV DNN with models like SSD-MobileNet is also an option.
    *   **Optimization:** INT8 quantization is crucial for performance. Low input resolutions (e.g., 160x160, 224x224) are recommended. Avoid full TensorFlow/PyTorch on the Pi.
2.  **`Synopsis.txt` - Core Project Goals:**
    *   AI-driven weed detection (CNNs).
    *   Precise elimination (spraying).
    *   Affordable and adaptable system.
    *   Rover halts, sprays weed, then resumes.
3.  **Guiding Principle:** Every software design choice, library selection, and model development step *must* be evaluated against these constraints and goals. If a component from the reference codebase doesn't fit (e.g., uses a heavy PyTorch model), you'll need to adapt or replace it.
