import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
import threading

# load
model = joblib.load("gesture_recognition_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_drawing = mp.solutions.drawing_utils

# ---------------- Voice control state ----------------
VOICE_ENABLED = True

# button area (x1,y1,x2,y2)
VOICE_BTN = (450, 20, 620, 70)

def speak_text(text, rate=155, volume=1.0):
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)
    engine.setProperty('volume', volume)
    engine.say(text)
    engine.runAndWait()

def speak_in_background(text):
    if not VOICE_ENABLED:
        return
    t = threading.Thread(target=lambda: speak_text(text))
    t.daemon = True
    t.start()

# ---------------- Feature extraction ----------------
def extract_for_predict(all_hands):
    features_list = []

    for lm in all_hands:
        pts = np.array([[p.x, p.y, p.z] for p in lm.landmark])

        wrist = pts[0]
        rel = (pts - wrist).flatten()
        dists = [np.linalg.norm(pts[i] - wrist) for i in [4,8,12,16,20]]

        vecs = [pts[i] - wrist for i in [4,8,12,16,20]]
        angs = []
        for i in range(len(vecs)-1):
            cos_val = np.dot(vecs[i], vecs[i+1]) / (
                np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[i+1]) + 1e-9
            )
            angs.append(np.arccos(np.clip(cos_val, -1, 1)))

        features_list.append(np.concatenate([rel, dists, angs]))

    if len(features_list) == 0:
        return np.zeros(144)
    elif len(features_list) == 1:
        return np.concatenate([features_list[0], np.zeros_like(features_list[0])])
    else:
        return np.concatenate([features_list[0], features_list[1]])

# ---------------- UI helpers ----------------
def draw_prediction_box(img, text, conf):
    cv2.rectangle(img, (20,20), (420,110), (0,0,0), -1)
    cv2.putText(img, f"Gesture: {text}", (30,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    cv2.putText(img, f"Conf: {conf:.2f}", (30,95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

def draw_voice_button(img):
    global VOICE_ENABLED
    x1,y1,x2,y2 = VOICE_BTN
    color = (0,200,0) if VOICE_ENABLED else (0,0,200)
    cv2.rectangle(img, (x1,y1), (x2,y2), color, -1)
    text = "VOICE: ON" if VOICE_ENABLED else "VOICE: OFF"
    cv2.putText(img, text, (x1+15, y1+35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

def mouse_callback(event, x, y, flags, param):
    global VOICE_ENABLED
    if event == cv2.EVENT_LBUTTONDOWN:
        x1,y1,x2,y2 = VOICE_BTN
        if x1 <= x <= x2 and y1 <= y <= y2:
            VOICE_ENABLED = not VOICE_ENABLED
            print("Voice:", "ON" if VOICE_ENABLED else "OFF")

# ---------------- Main loop ----------------
def main():
    cap = cv2.VideoCapture(0)
    last_pred = None

    cv2.namedWindow("Gesture Recognition (Live)")
    cv2.setMouseCallback("Gesture Recognition (Live)", mouse_callback)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        img = cv2.cvtColor(cv2.flip(frame,1), cv2.COLOR_BGR2RGB)
        results = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        all_hands = []
        if results.multi_hand_landmarks:
            left, right = None, None
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = results.multi_handedness[idx].classification[0].label
                if label == "Left":
                    left = hand_landmarks
                else:
                    right = hand_landmarks

            if left is not None:
                all_hands.append(left)
                mp_drawing.draw_landmarks(img, left, mp_hands.HAND_CONNECTIONS)
            if right is not None:
                all_hands.append(right)
                mp_drawing.draw_landmarks(img, right, mp_hands.HAND_CONNECTIONS)

        feats = extract_for_predict(all_hands).reshape(1, -1)

        if len(all_hands) > 0:
            proba = model.predict_proba(feats)[0]
            idx = np.argmax(proba)
            conf = proba[idx]

            if conf > 0.65:
                pred = label_encoder.inverse_transform([idx])[0]
                draw_prediction_box(img, pred, conf)

                if pred != last_pred:
                    speak_in_background(pred)
                    last_pred = pred
        else:
            last_pred = None

        draw_voice_button(img)

        cv2.imshow("Gesture Recognition (Live)", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
