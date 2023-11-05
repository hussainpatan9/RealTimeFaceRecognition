import cv2
import face_recognition
import logging

NUM_CAMERAS = 5
REFERENCE_IMAGES = {
    "Person1": "path_to_person1_image.jpg",
    "Person2": "path_to_person2_image.jpg",
    "Person3": "path_to_person3_image.jpg",
    # Add more persons as needed
}

LOG_FILE = "face_recognition.log"
LOG_LEVEL = logging.INFO

DELAY_BETWEEN_FRAMES = 1  # Adjust as needed


def setup_logging():
    logging.basicConfig(filename=LOG_FILE, level=LOG_LEVEL)


def recognize_faces(frame, reference_encodings):
    face_locations, face_encodings = get_face_data(frame)

    for (top, right, bottom, left), face_encoding in zip(
            face_locations, face_encodings):
        name = get_person_name(face_encoding, reference_encodings)

        draw_face_rectangle(frame, top, right, bottom, left)
        draw_person_name(frame, name, left + 6, bottom - 6)

    return frame


def get_face_data(frame):
    # Use the more accurate "cnn" model for face locations
    face_locations = face_recognition.face_locations(frame, model="hog")
    # Increase num_jitters for fine-tuning face encodings
    face_encodings = face_recognition.face_encodings(
        frame, face_locations, num_jitters=2)
    return face_locations, face_encodings


def get_person_name(face_encoding, reference_encodings):
    matches = face_recognition.compare_faces(
        list(reference_encodings.values()), face_encoding
    )

    for match, person_name in zip(matches, reference_encodings.keys()):
        if match:
            return person_name
    return "Unknown"


def draw_face_rectangle(frame, top, right, bottom, left):
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)


def draw_person_name(frame, name, x, y):
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, name, (x, y), font, 0.5, (255, 255, 255), 1)


def load_reference_images(reference_images):
    reference_encodings = {}
    for name, image_path in reference_images.items():
        try:
            reference_image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(
                reference_image, num_jitters=5)[0]
            reference_encodings[name] = encoding
        except (FileNotFoundError, IndexError) as e:
            logging.error(f"Error processing {name}'s image: {e}")
    return reference_encodings


def list_available_cameras():
    available_cameras = [
        i for i in range(NUM_CAMERAS) if cv2.VideoCapture(i).isOpened()
    ]
    return available_cameras


def select_camera():
    available_cameras = list_available_cameras()

    if not available_cameras:
        print("No cameras found.")
        logging.error("No cameras found.")
        return None

    logging.info("Available cameras:")
    for idx in available_cameras:
        print(f"Camera {idx}")
        logging.info(f"Camera {idx}")

    selected_camera = None
    while selected_camera is None:
        try:
            selected_camera = int(
                input("Select the camera (enter the camera index): "))
            if selected_camera not in available_cameras:
                logging.error(
                    "Invalid camera selection. Please choose a valid camera.")
                selected_camera = None
        except ValueError:
            logging.error("Invalid input. Please enter a valid camera index.")
            selected_camera = None

    return selected_camera


def main():
    setup_logging()

    selected_camera = select_camera()
    if selected_camera is None:
        return

    cap = cv2.VideoCapture(selected_camera)

    if not cap.isOpened():
        logging.error("Error opening the selected camera.")
        return

    try:
        reference_encodings = load_reference_images(REFERENCE_IMAGES)

        while True:
            ret, frame = cap.read()

            if not ret:
                logging.error("Error reading frame from the camera.")
                break

            frame = recognize_faces(frame, reference_encodings)

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(DELAY_BETWEEN_FRAMES) & 0xFF == ord("q"):
                break
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
