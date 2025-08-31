# hand_gesture.py
# Real-time hand landmarks via MediaPipe; simple rule-based recognition (thumbs up/down, fist/open)
# Run: python hand_gesture.py

import cv2, mediapipe as mp, numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def classify(landmarks):
    # landmarks: list of 21 (x,y) normalized points. Very simple heuristics:
    tips = [4,8,12,16,20]  # thumb, index, middle, ring, pinky
    folded = 0
    for t in tips[1:]:
        if landmarks[t].y > landmarks[t-2].y: folded += 1
    # if most fingers folded -> fist
    if folded >= 4: return "Fist"
    if landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y:
        return "Two Fingers Up"
    # thumb up logic (approx)
    if landmarks[4].x < landmarks[3].x: return "Thumbs Up"
    return "Open Hand"

def main():
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret: break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(img)
            if res.multi_hand_landmarks:
                for handLms in res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                    lm = handLms.landmark
                    label = classify(lm)
                    cv2.putText(frame, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.imshow("Hand Gesture", frame)
            if cv2.waitKey(1)&0xFF==27: break
    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    main()
