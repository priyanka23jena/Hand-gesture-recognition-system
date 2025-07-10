import copy
import argparse
import cv2 as cv
import mediapipe as mp
from collections import deque, Counter

# Argument parsing
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    return parser.parse_args()

# Gesture classifier using landmark positions
class KeyPointClassifier:
    def __call__(self, landmarks):
        mp_hands = mp.solutions.hands

        def is_finger_extended(tip_id, pip_id, margin=0.02):
            return (landmarks.landmark[tip_id].y + margin) < landmarks.landmark[pip_id].y

        # Check fingers (index=8, middle=12, ring=16, pinky=20)
        fingers = [is_finger_extended(8, 6),
                   is_finger_extended(12, 10),
                   is_finger_extended(16, 14),
                   is_finger_extended(20, 18)]

        # Thumb check (left to right for right hand)
        thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
        thumb_extended = (thumb_tip.x + 0.02) < thumb_ip.x  # Added margin to avoid jitter

        if all(fingers) and not thumb_extended:
            return 0  # Open Hand
        elif fingers[0] and fingers[1] and not any(fingers[2:]) and not thumb_extended:
            return 1  # Peace
        elif not any(fingers) and not thumb_extended:
            return 2  # Closed Fist
        elif all(fingers[:3]) and not fingers[3]:
            return 3  # Waving (3 fingers)
        elif thumb_extended and fingers[3] and not any(fingers[:3]):
            return 4  # Yo
        elif (thumb_tip.y + 0.02) < thumb_ip.y and not any(fingers):
            return 5  # Thumbs Up
        else:
            return -1  # No Gesture

def main():
    args = get_args()

    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=args.use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence
    )

    mp_drawing = mp.solutions.drawing_utils
    keypoint_classifier = KeyPointClassifier()

    gesture_labels = ["Open Hand", "Peace", "Closed Fist", "Waving", "Yo", "Thumbs Up"]

    gesture_history = deque(maxlen=5)  # buffer for last 5 detections

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        debug_image = copy.deepcopy(frame)

        rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        results = hands.process(rgb_image)
        rgb_image.flags.writeable = True

        gesture_id = -1

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(debug_image, landmarks, mp_hands.HAND_CONNECTIONS)
                gesture_id = keypoint_classifier(landmarks)
                gesture_history.append(gesture_id)

        if gesture_history:
            most_common_id, count = Counter(gesture_history).most_common(1)[0]
            gesture_text = gesture_labels[most_common_id] if most_common_id != -1 else "No Gesture"
        else:
            gesture_text = "No Gesture"

        cv.putText(debug_image, f"Gesture: {gesture_text}", (20, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv.imshow("Hand Gesture Recognition", debug_image)

        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
