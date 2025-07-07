import cv2
import mediapipe as mp

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam.")
    exit()

# Load face detector
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Finger landmark tips (thumb is 4, others are 8, 12, 16, 20)
FINGER_TIPS = [4, 8, 12, 16, 20]

def get_hand_gesture(hand_landmarks):
    finger_up = []

    # Get landmark coordinates of wrist and fingers
    landmarks = hand_landmarks.landmark

    # Thumb: check if tip is to the right of joint (for right hand) or left (for left hand)
    if landmarks[FINGER_TIPS[0]].x < landmarks[FINGER_TIPS[0] - 1].x:
        finger_up.append(1)  # Thumb up
    else:
        finger_up.append(0)

    # Other fingers: check if tip is above joint
    for tip_id in FINGER_TIPS[1:]:
        if landmarks[tip_id].y < landmarks[tip_id - 2].y:
            finger_up.append(1)
        else:
            finger_up.append(0)

    # Classify gestures based on finger patterns
    if finger_up == [0, 0, 0, 0, 0]:
        return "Fist ✊"
    elif finger_up == [1, 1, 1, 1, 1]:
        return "Open Palm ✋"
    elif finger_up == [1, 0, 0, 0, 0]:
        return "Thumbs Up 👍"
    elif finger_up == [0, 1, 1, 0, 0]:
        return "Peace ✌️"
    else:
        return "Unknown Gesture"

# Create window
window_name = "Face + Hand Detection +Gesture Recognition"
cv2.namedWindow(window_name)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror view
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Face Detection ---
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # --- Hand Detection ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            gesture = get_hand_gesture(hand_landmarks)
            h, w, _ = frame.shape
            cx = int(hand_landmarks.landmark[0].x * w)
            cy = int(hand_landmarks.landmark[0].y * h)
            cv2.putText(frame, gesture, (cx - 50, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1)
    if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
