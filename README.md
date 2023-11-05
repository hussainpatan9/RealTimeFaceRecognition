
# Real-Time Face Recognition

This Python script uses OpenCV and face_recognition to perform real-time face recognition using a webcam. It detects faces in the camera feed, compares them with reference images, and labels recognized faces with their names.

## Prerequisites

Ensure you have the required Python libraries installed:

```bash
pip install opencv-python face-recognition
```

## Usage

1. Run the script:

```bash
python face_recognition_script.py
```

2. Select the camera when prompted by entering the camera index.

3. The script will open a window displaying real-time face recognition.

4. Press 'q' to exit the script.

## Configuration

- `NUM_CAMERAS`: Number of available cameras for selection.
- `REFERENCE_IMAGES`: Dictionary with reference names and image paths for face recognition.
- `LOG_FILE`: File for error and information logging.
- `LOG_LEVEL`: Logging level (default: INFO).
- `DELAY_BETWEEN_FRAMES`: Delay in milliseconds between processed frames.

## Additional Notes

- The script uses the "hog" model for face locations and a reduced number of jitters for faster processing. Adjust parameters based on your requirements.

- Ensure that the required reference images are provided in the `REFERENCE_IMAGES` dictionary.

- Press 'q' to exit the script.

## License

This project is licensed under the [MIT License](LICENSE).