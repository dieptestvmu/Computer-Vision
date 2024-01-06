# Eye Tracking Mouse Control

## Introduction
This software is developed to detect the eye gaze direction and utilize that information to control the mouse on a computer. The application is capable of recognizing actions such as up, down, left, right, center, and eye blink.

![Eye Tracking Mouse Control](https://i.ibb.co/YjTLstK/Demo.png)

## Requirements
- **Programming language:** Python
- **Libraries:** OpenCV, Tensorflow, Mediapipe, Numpy, ...

## Usage
1. **Install the necessary libraries:**
    ```bash
    pip install opencv-python tensorflow mediapipe numpy
    ```

2. **Create and train the models:**
    - Prepare the dataset following the structure:
        ```
        Dataset/
        ├── left_eye/
        |   ├── up/
        |   ├── down/
        |   ├── left/
        |   ├── right/
        |   ├── center/
        |   ├── blink/
        ├── right_eye/
            ├── blink/
            ├── noblink/
        ```
    - Use the training script to create two models: `model_left_eye` and `model_right_eye`.

3. **Run the program:**
    ```bash
    python eye_tracking_mouse.py
    ```

## Project Structure
- `eye_tracking_mouse.py`: The main program that performs eye gaze detection and mouse control.
- `train.py`: Script for training the models from the dataset.
- `models/`: Directory containing the trained models.
- `Dataset/`: Directory containing the training dataset.

## Support and Contribution
If you encounter issues or have suggestions for improvement, please open an issue on GitHub. We welcome contributions from the community.

## License
This project is distributed under the [MIT License](LICENSE).
