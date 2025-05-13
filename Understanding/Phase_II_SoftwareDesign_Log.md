# II. Software Design for Deployability and Control

**A. Architectural Considerations**

A modular, layered architecture is essential for managing complexity, testing, and hardware interaction.

1.  **Layered Architecture:**
    *   **Application Layer:** Orchestrates the overall workflow (e.g., main control loop, system states).
    *   **Core Logic Layer:**
        *   **AI Inference Module:** Handles loading the TFLite model, image preprocessing, running inference, and post-processing results.
        *   **Decision Module:** Interprets model output to classify plants and decide actions (e.g., "weed detected," "spray," "move forward").
    *   **Hardware Abstraction Layer (HAL):**
        *   **Camera Module:** Interface for capturing images from the Pi camera.
        *   **Motion Control Module:** Interface for controlling the rover's chassis (e.g., `move_forward()`, `stop()`). This will likely communicate with an Arduino or a motor driver.
        *   **Sprayer Control Module:** Interface for activating/deactivating the spraying mechanism (e.g., `activate_sprayer()`, `deactivate_sprayer()`).
    *   **Communication Module (Optional but Recommended):** If using an Arduino for motor/sprayer control, this module handles serial communication between the Raspberry Pi and Arduino. Define a simple, robust protocol (e.g., command strings like "MOVE:FWD", "SPRAY:ON").

**B. Designing for Modularity and Adherence to `restrictions.txt`**

*   **Clear Interfaces:** Define Python classes or modules for each component in the HAL (e.g., `ChassisController`, `SprayerActuator`, `PiCamera`). The core logic will use these well-defined APIs.
*   **Decoupling:** The AI and decision logic should not have direct knowledge of low-level hardware details (GPIO pins, serial commands). This allows you to:
    *   Test core logic on your development PC by "mocking" hardware responses.
    *   Easily adapt to changes in hardware (e.g., different motor driver) by only modifying the HAL.
    *   Ensure HAL implementations are optimized for the Pi Zero 2 W as per `restrictions.txt`.
*   **Configuration Files:** Use configuration files (e.g., JSON, YAML) for settings like camera resolution, model paths, serial port details, spray duration, etc., rather than hardcoding.
