import cv2
import numpy as np
import mediapipe as mp
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Video Capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logging.error("Camera could not be opened. Check your webcam.")
    exit()

# Prompt for data label
name = input("Enter the name of the data: ").strip()
if not name:
    logging.error("Invalid input: name cannot be empty.")
    cap.release()
    exit()

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(static_image_mode=False)

X = []
frame_count = 0
MAX_FRAMES = 100

def extract_landmark_list(results):
    lst = []

    # Normalize face
    if results.face_landmarks:
        base_x = results.face_landmarks.landmark[1].x
        base_y = results.face_landmarks.landmark[1].y
        for lm in results.face_landmarks.landmark:
            lst.append(lm.x - base_x)
            lst.append(lm.y - base_y)
    else:
        return None  # skip frame if no face detected

    # Normalize left hand
    if results.left_hand_landmarks:
        base_x = results.left_hand_landmarks.landmark[8].x
        base_y = results.left_hand_landmarks.landmark[8].y
        for lm in results.left_hand_landmarks.landmark:
            lst.append(lm.x - base_x)
            lst.append(lm.y - base_y)
    else:
        lst.extend([0.0] * 42)

    # Normalize right hand
    if results.right_hand_landmarks:
        base_x = results.right_hand_landmarks.landmark[8].x
        base_y = results.right_hand_landmarks.landmark[8].y
        for lm in results.right_hand_landmarks.landmark:
            lst.append(lm.x - base_x)
            lst.append(lm.y - base_y)
    else:
        lst.extend([0.0] * 42)

    return lst

logging.info(f"Starting data collection for '{name}'...")

while frame_count < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        logging.warning("Frame not read properly. Skipping.")
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

    landmarks = extract_landmark_list(results)
    if landmarks:
        X.append(landmarks)
        frame_count += 1

    # Draw landmarks
    mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Overlay frame count
    cv2.putText(frame, f"Frames: {frame_count}/{MAX_FRAMES}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show window
    cv2.imshow("Data Collection", frame)

    if cv2.waitKey(1) == 27:  # ESC key
        logging.info("ESC pressed. Exiting early.")
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
holistic.close()

# Save data
X_array = np.array(X)
np.save(f"{name}.npy", X_array)
logging.info(f"Data saved to '{name}.npy'. Shape: {X_array.shape}")
