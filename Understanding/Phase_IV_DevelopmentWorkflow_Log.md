## IV. Development Workflow and Next Steps

1.  **Phase 1: Setup & Baseline (1-2 weeks)**
    *   **Action:** Set up the Raspberry Pi Zero 2 W (OS, Python, TFLite, OpenCV, camera drivers).
    *   **Action:** Clone/transfer the `AgroBot_Reference` codebase and dataset.
    *   **Action:** Get a very basic TFLite model (either converted from reference or a standard one) running inference on static images on the Pi. Measure initial inference time.
    *   **Action:** Implement basic image capture from the Pi camera.
2.  **Phase 2: Data Pipeline & Initial Model Training (2-3 weeks)**
    *   **Action:** Organize your full dataset (reference + any initial new data).
    *   **Action:** Develop and test your data augmentation pipeline.
    *   **Action:** Set up a training environment (can be your PC). Train/fine-tune your chosen model architecture.
    *   **Action:** Convert to TFLite, apply INT8 quantization, and evaluate accuracy/speed on the Pi.
3.  **Phase 3: Hardware Abstraction Layer (HAL) for Motion & Spraying (2-3 weeks)**
    *   **Action (Parallel to Phase 2 if team):** If using Arduino, develop and test the Arduino sketch for motor/sprayer control.
    *   **Action:** Develop the Python HAL modules on the Pi for `ChassisController` and `SprayerActuator`.
    *   **Action:** Test communication and control (e.g., make motors spin, turn sprayer relay on/off via Python commands).
4.  **Phase 4: Integration - AI with Control (2-3 weeks)**
    *   **Action:** Develop the main application logic on the Pi that:
        1.  Captures image.
        2.  Runs AI inference.
        3.  Makes a decision (weed/no weed).
        4.  Commands chassis to stop/move.
        5.  Commands sprayer if weed detected.
    *   **Action:** Begin integration testing in a controlled environment.
5.  **Phase 5: Field Testing, Iteration & Enhancement (Ongoing)**
    *   **Action:** Test the AgroBot in increasingly realistic field conditions.
    *   **Action:** Continuously collect new data, especially for scenarios where the model performs poorly.
    *   **Action:** Regularly retrain, re-quantize, and re-evaluate your model.
    *   **Action:** Refine software based on field observations (e.g., timing, robustness).

**Integrating New Data:**
*   Establish a routine: Collect -> Annotate -> Add to training set -> Retrain -> Evaluate -> Deploy.
*   Version your datasets and models.

**Testing Practices:**
*   **Unit Tests:** For pure Python logic (e.g., data parsing, decision rules).
*   **Integration Tests:** For interactions (Pi-camera, Pi-AI model, Pi-Arduino).
*   **End-to-End (E2E) Field Tests:** The ultimate validation. Test the complete weed detection and spraying cycle in the target environment. Record successes and failures for analysis.
