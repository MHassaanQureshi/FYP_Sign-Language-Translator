import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
import threading
from collections import deque

# ================= LOAD MODEL =================
model = joblib.load("gesture_recognition_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_drawing = mp.solutions.drawing_utils

# ================= CONFIG =================
VOICE_ENABLED = True
CONF_THRESHOLD = 0.70
ENTROPY_THRESHOLD = 1.4
SMOOTHING_FRAMES = 7

VOICE_BTN = (450, 20, 620, 70)

# ================= VOICE =================
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

# ================= FEATURE EXTRACTION =================
def extract_for_predict(all_hands):
    features = []

    for lm in all_hands:
        pts = np.array([[p.x, p.y, p.z] for p in lm.landmark])

        wrist = pts[0]
        rel = (pts - wrist).flatten()

        tips = [4, 8, 12, 16, 20]
        dists = [np.linalg.norm(pts[i] - wrist) for i in tips]

        vecs = [pts[i] - wrist for i in tips]
        angs = []
        for i in range(len(vecs) - 1):
            cos_val = np.dot(vecs[i], vecs[i + 1]) / (
                np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[i + 1]) + 1e-9
            )
            angs.append(np.arccos(np.clip(cos_val, -1, 1)))

        features.append(np.concatenate([rel, dists, angs]))

    if len(features) == 0:
        return np.zeros(144)
    elif len(features) == 1:
        return np.concatenate([features[0], np.zeros_like(features[0])])
    else:
        return np.concatenate([features[0], features[1]])

# ================= UTILS =================
def prediction_entropy(probs):
    probs = np.clip(probs, 1e-9, 1.0)
    return -np.sum(probs * np.log(probs))

def draw_prediction_box(img, text, conf):
    color = (0, 255, 0) if text != "UNKNOWN" else (0, 0, 255)
    cv2.rectangle(img, (20, 20), (420, 110), (0, 0, 0), -1)
    cv2.putText(img, f"Gesture: {text}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    if text != "UNKNOWN":
        cv2.putText(img, f"Conf: {conf:.2f}", (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def draw_voice_button(img):
    color = (0, 200, 0) if VOICE_ENABLED else (0, 0, 200)
    x1, y1, x2, y2 = VOICE_BTN
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    text = "VOICE: ON" if VOICE_ENABLED else "VOICE: OFF"
    cv2.putText(img, text, (x1 + 15, y1 + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def mouse_callback(event, x, y, flags, param):
    global VOICE_ENABLED
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1, x2, y2 = VOICE_BTN
        if x1 <= x <= x2 and y1 <= y <= y2:
            VOICE_ENABLED = not VOICE_ENABLED
            print("Voice:", "ON" if VOICE_ENABLED else "OFF")

# ================= MAIN =================
def main():
    cap = cv2.VideoCapture(0)
    last_spoken = None
    pred_buffer = deque(maxlen=SMOOTHING_FRAMES)

    cv2.namedWindow("Gesture Recognition (Live)")
    cv2.setMouseCallback("Gesture Recognition (Live)", mouse_callback)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        img = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        all_hands = []
        if results.multi_hand_landmarks:
            for idx, hand_lm in enumerate(results.multi_hand_landmarks):
                all_hands.append(hand_lm)
                mp_drawing.draw_landmarks(
                    img, hand_lm, mp_hands.HAND_CONNECTIONS
                )

        feats = extract_for_predict(all_hands).reshape(1, -1)

        if len(all_hands) > 0:
            proba = model.predict_proba(feats)[0]
            idx = np.argmax(proba)
            conf = proba[idx]
            entropy = prediction_entropy(proba)

            if conf > CONF_THRESHOLD and entropy < ENTROPY_THRESHOLD:
                pred = label_encoder.inverse_transform([idx])[0]
            else:
                pred = "UNKNOWN"
                conf = 0.0
        else:
            pred = "UNKNOWN"
            conf = 0.0

        pred_buffer.append(pred)
        final_pred = max(set(pred_buffer), key=pred_buffer.count)

        draw_prediction_box(img, final_pred, conf)

        if final_pred != "UNKNOWN" and final_pred != last_spoken:
            speak_in_background(final_pred)
            last_spoken = final_pred

        if final_pred == "UNKNOWN":
            last_spoken = None

        draw_voice_button(img)
        cv2.imshow("Gesture Recognition (Live)", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
