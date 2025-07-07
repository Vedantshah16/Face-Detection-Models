import cv2
import mediapipe as mp
import datetime
import screen_brightness_control as sbc

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam.")
    exit()

# Face detector
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# MediaPipe hand setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

FINGER_TIPS = [4, 8, 12, 16, 20]
brightness = sbc.get_brightness()[0]  # Get current brightness

def get_hand_gesture(landmarks):
    finger_up = []
    # Thumb
    if landmarks[FINGER_TIPS[0]].x < landmarks[FINGER_TIPS[0] - 1].x:
        finger_up.append(1)
    else:
        finger_up.append(0)
    # Other fingers
    for tip_id in FINGER_TIPS[1:]:
        if landmarks[tip_id].y < landmarks[tip_id - 2].y:
            finger_up.append(1)
        else:
            finger_up.append(0)

    if finger_up == [0, 0, 0, 0, 0]:
        return "Fist ✊"
    elif finger_up == [1, 1, 1, 1, 1]:
        return "Open Palm ✋"
    elif finger_up == [1, 0, 0, 0, 0]:
        return "Thumbs Up 👍"
    elif finger_up == [0, 1, 1, 0, 0]:
        return "Peace ✌️"
    else:
        return "Unknown"

def save_screenshot(frame):
    filename = f"screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(filename, frame)
    print(f"📸 Screenshot saved: {filename}")

def adjust_brightness(change):
    global brightness
    brightness = max(10, min(100, brightness + change))  # Clamp between 10–100
    sbc.set_brightness(brightness)
    print(f"💡 Brightness set to: {brightness}%")

# Run
window_name = "Gesture Detection with Brightness & Screenshot"
cv2.namedWindow(window_name)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face Detection
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Hand Detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = get_hand_gesture(hand_landmarks.landmark)

            # Get center for text
            h, w, _ = frame.shape
            cx = int(hand_landmarks.landmark[0].x * w)
            cy = int(hand_landmarks.landmark[0].y * h)
            cv2.putText(frame, gesture, (cx - 60, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # ✌️ Save Screenshot
            if gesture == "Peace ✌️":
                save_screenshot(frame)

            # ✋ Increase Brightness
            elif gesture == "Open Palm ✋":
                adjust_brightness(+10)

            # ✊ Decrease Brightness
            elif gesture == "Fist ✊":
                adjust_brightness(-10)

    # Show frame
    cv2.imshow(window_name, frame)

    # Exit on 'q' or window close
    key = cv2.waitKey(1)
    if key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
