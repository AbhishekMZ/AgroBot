# III. Model Accuracy and Performance Strategy

## A. Systematic Approach to Improving Model Accuracy (Data-Centric)

1.  **Establish a Baseline:**
    *   If the reference model can be converted to TFLite and run on the Pi, benchmark its accuracy and speed.
    *   Otherwise, start with a simple, known-to-work TFLite model from `restrictions.txt` (e.g., MobileNetV2-SSD Lite) trained on a subset of your data.
    *   Key Metrics: Precision, Recall, F1-score (for weed class), mAP (if object detection), inference time (ms per frame on Pi).
2.  **Iterative Data Enhancement:**
    *   **Leverage `AgroBot_Reference` Data:** Use the dataset from the reference repository as your starting point.
    *   **Collect Diverse, Representative Data:** This is often the most impactful way to improve accuracy. Capture images under various conditions:
        *   Different lighting (sunny, cloudy, shadows).
        *   Different times of day.
        *   Various crop growth stages and weed types/sizes.
        *   Different soil types and moisture levels.
    *   **High-Quality Annotations:** Ensure labels are accurate and consistent.
    *   **Data Augmentation:** Apply aggressive augmentation suitable for plants (flips, rotations, scaling, brightness, contrast, saturation, hue shifts, blur, noise). Use libraries like `Albumentations`.
3.  **Model Training and Optimization (respecting `restrictions.txt`):**
    *   **Transfer Learning:** Fine-tune pre-trained TFLite-compatible backbones (e.g., MobileNet, EfficientDet-Lite backbones) on your dataset. This is usually more effective than training from scratch with limited data.
    *   **Hyperparameter Tuning:** Experiment with learning rate, batch size, optimizer, and regularization.
    *   **Model Architecture (if necessary):** If standard models aren't sufficient and you have the expertise, explore designing custom lightweight CNNs, but always prioritize TFLite compatibility and performance on the Pi.
    *   **Quantization:** This is **critical**.
        *   Start with post-training dynamic range quantization.
        *   Move to **post-training INT8 quantization**. This requires a representative dataset (a few hundred samples) for calibration. It significantly speeds up inference and reduces model size.
4.  **Error Analysis:** Regularly analyze your model's mistakes. Are there specific types of weeds or conditions it struggles with? Use this insight to guide further data collection or model adjustments.

**B. Benchmarking and Tracking Performance**

*   **On-Target Benchmarking:** Always measure inference speed and accuracy **on the Raspberry Pi Zero 2 W**.
    *   Use `tflite_benchmark_tool` for standardized speed tests.
    *   Write scripts to evaluate accuracy on a test set directly on the Pi.
*   **Experiment Logging:** Keep a meticulous record of:
    *   Dataset version used.
    *   Augmentation techniques applied.
    *   Model architecture and hyperparameters.
    *   Quantization method.
    *   Resulting accuracy metrics and inference speed on Pi.
    *   (Simple CSV/spreadsheet is fine; tools like MLflow are options for larger projects).

**C. Overall Software Performance Optimization (for Pi Zero 2 W)**

*   **Efficient Python Code:**
    *   Use optimized libraries like NumPy for array operations.
    *   Profile your Python code (`cProfile`) on the Pi to find bottlenecks.
    *   Minimize memory allocations and data copying.
*   **Image Processing Pipeline:** Optimize image capture, resizing, and normalization steps (OpenCV is generally efficient).
*   **Threading/Async (Cautiously):** The Pi Zero 2 W has a quad-core CPU. You might use threading for I/O-bound tasks (like waiting for camera frame or serial response) to prevent blocking the main AI processing loop. Be mindful of the Global Interpreter Lock (GIL) for CPU-bound tasks in Python. `asyncio` can also be an option.
*   **Lean System:** Keep background processes on the Pi to a minimum.
