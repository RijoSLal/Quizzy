import cv2
import threading
import numpy as np
from keras.applications.mobilenet_v3 import preprocess_input # type: ignore
import mlflow
import mediapipe as mp
import logging

logger = logging.getLogger("interview")

mlflow.set_tracking_uri("https://dagshub.com/slalrijo2005/Quizzy.mlflow")

# load the trained model
model_name = "mobilenet_model"
model_version = 4
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.tensorflow.load_model(model_uri)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5)


class VideoCamera(object):

    """
    This class uses a vision model to predict emotions from live webcam video.
    It implements a singleton pattern to ensure only one instance of the camera is used.
    """
    _instance = None  # single instance singleton used because to access video cam in single obj

    def __new__(cls, *args, **kwargs):
        
        """
        Ensures only a single instance of VideoCamera is created.
        Initializes the camera on the first instantiation.
        """

        if cls._instance is None:
            cls._instance = super(VideoCamera, cls).__new__(cls)
            # cls._instance._init_camera()
        return cls._instance
    
    def __init__(self):
        """
        Initializes the video capture, validates camera accessibility,
        and starts a separate thread for continuous frame capture.

        """
        if hasattr(self, "video"):  # Prevent re-initialization
            return

        self.video = cv2.VideoCapture(0)
        if not self.video.isOpened():
            logger.error("Camera cannot be accessed")
            raise RuntimeError("Camera could not be opened!")
            
            
        self.grabbed, self.frame = self.video.read()
        if self.frame is None:
            logger.error("Frames cannot be accessed")
            raise RuntimeError("No frame captured! Check camera connection.")
        
        self.running = True
        self.counts = [0] * 3  # Counts for [0, 1, 2]
        self.total = 0  # Total frames processed
        self.pos_count = 0  # Count occurrences where pos is True
        self.p = [0, 0, 0, 0]  # Probabilities
        threading.Thread(target=self.update, daemon=True).start()
    
    def release(self)-> None:
        """
        Releases the video capture object and stops the camera.
        """
        self.running = False
        self.video.release() #to stop camera

    def reset_updates(self) -> None:
        """
        Resets the updated probabilities
        """
        self.counts = [0] * 3  # Counts for [0, 1, 2]
        self.total = 0  # Total frames processed
        self.pos_count = 0  # Count occurrences where pos is True
        self.p = [0, 0, 0, 0]  # Probabilities

    def detect_head_down(self, nose: tuple[int, int], left_eye: tuple[int, int], right_eye: tuple[int, int]) -> bool:
        """
        Detects whether the head is tilted downward based on facial landmarks.

        Args:
            nose (tuple[int, int]): Coordinates of the nose tip.
            left_eye (tuple[int, int]): Coordinates of the left eye.
            right_eye (tuple[int, int]): Coordinates of the right eye.

        Returns:
            bool: True if the head is down, otherwise False.
        """
        mid_eye_y = (left_eye[1] + right_eye[1]) // 2
        return nose[1] > mid_eye_y - 6 

    def get_landmark(self, face_landmarks, lm_id: int, w: int, h: int) ->  tuple[int, int]:
        """
        Extracts a specific facial landmark's coordinates.

        Args:
            face_landmarks: Mediapipe face landmarks object.
            lm_id (int): Landmark ID.
            w (int): Frame width.
            h (int): Frame height.

        Returns:
            tuple[int, int]: (x, y) coordinates of the landmark.
        """
        lm = face_landmarks.landmark[lm_id]
        return int(lm.x * w), int(lm.y * h)

    def get_frame(self) -> bytes:
        """
        Captures a frame, processes facial landmarks, detects emotions and head position,
        and encodes the frame into JPEG format.

        Returns:
            bytes: Encoded JPEG image bytes.
        """
        image = self.frame
        h, w, _ = image.shape  # Frame dimensions
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Get key facial landmarks
                    left_eye : tuple[int, int] = self.get_landmark(face_landmarks, 33, w, h)
                    right_eye : tuple[int, int]= self.get_landmark(face_landmarks, 263, w, h)
                    nose_tip : tuple[int, int] = self.get_landmark(face_landmarks, 8, w, h)

                    # Detect head position
                    head_position : bool = self.detect_head_down(nose_tip, left_eye, right_eye)

                    # Get face bounding box
                    x_min, y_min, x_max, y_max = w, h, 0, 0
                    for lm in face_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        x_min, y_min = min(x_min, x), min(y_min, y)
                        x_max, y_max = max(x_max, x), max(y_max, y)

                    # Ensure valid cropping
                    x_min, y_min = max(0, x_min - 10), max(0, y_min - 10)
                    x_max, y_max = min(w, x_max + 10), min(h, y_max + 10)

                    # Crop face region
                    face_only = image[y_min:y_max, x_min:x_max]

                    try:
                        face_resized = cv2.resize(face_only, (224, 224))
                        img = preprocess_input(face_resized.astype(np.float32))
                        img = np.expand_dims(img, axis=0)

                        prediction = model.predict(img)
                        predicted_class = np.argmax(prediction)
                        self.upgrade(predicted_class,head_position)
                    except Exception as e:
                        logger.error(f"Prediction error : {str(e)}",exc_info=True)

        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def upgrade(self, emo: int, pos: bool) ->None:
        """
        Updates emotion counts and calculates probability distribution.

        Args:
            emo (int): Predicted emotion class index.
            pos (bool): Boolean indicating head position (True if head is down).
        """
        self.counts[emo] += 1
        self.total += 1
        if pos:
            self.pos_count += 1  

        # update probabilities
        self.p = [round((count / self.total) * 100, 2) for count in self.counts]
        self.p.append(round((self.pos_count / self.total) * 100, 2))  # Append positive probability
        # logger.info(f"Updated probabilities: {self.p}")

    
    def get_latest_prediction(self)-> list[float]:
        """
        Returns the latest calculated probability distribution of emotions and head position.

        Returns:
            list[float]: List containing emotion probabilities and head position probability.
        """
        return self.p

    def update(self) -> None:
        """
        Continuously updates the latest frame from the video capture in a loop.
        """
        while True:
            (self.grabbed, self.frame) = self.video.read()

def generate_frames(camera:VideoCamera):
    """
    Continuously generates frames from the VideoCamera instance and formats them 
    for HTTP multipart streaming.

    Args:
        camera (VideoCamera): The instance of VideoCamera.

    Yields:
        bytes: Encoded JPEG frames in multipart format.
    """
    while True:
        try:
            frame: bytes = camera.get_frame()
            if frame is None:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        except Exception as e:
            logger.error(f"Frame generation error: {str(e)}")
            break  # break if there is an error

    camera.release() 

