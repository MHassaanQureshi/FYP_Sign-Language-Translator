import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
import threading

# ===== Load trained model + label encoder =====
model = joblib.load('gesture_recognition_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# ===== Mediapipe setup =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# ===== SPEECH =====
def speak_text(text, rate=155, volume=1.0):
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)
    engine.setProperty('volume', volume)
    engine.say(text)
    engine.runAndWait()

def speak_in_background(text):
    thread = threading.Thread(target=lambda: speak_text(text))
    thread.daemon = True
    thread.start()

# ===== FEATURE EXTRACTOR (Single + Double Hand Compatible) =====
# def extract_features_for_prediction(all_hands):
#     """
#     all_hands = list of mediapipe landmarks for each detected hand
#     Always returns 144 features (single or double hand)
#     """
#     features_list = []

#     for lm in all_hands:
#         pts = np.array([[p.x, p.y, p.z] for p in lm])
#         wrist = pts[0]
#         relative = pts - wrist

#         fingertips = [4, 8, 12, 16, 20]
#         distances = [np.linalg.norm(pts[i] - wrist) for i in fingertips]

#         vectors = [pts[i] - wrist for i in fingertips]
#         angles = []
#         for i in range(len(vectors) - 1):
#             cos_angle = np.dot(vectors[i], vectors[i + 1]) / (
#                 np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[i + 1])
#             )
#             angles.append(np.arccos(np.clip(cos_angle, -1, 1)))

#         features_list.append(np.concatenate([relative.flatten(), distances, angles]))

#     # Always ensure 2 hands features
#     if len(features_list) == 0:
#         return np.zeros(144)
#     elif len(features_list) == 1:
#         return np.concatenate([features_list[0], np.zeros_like(features_list[0])])
#     else:
#         return np.concatenate([features_list[0], features_list[1]])
def extract_features_for_prediction(all_hands):
    """
    Extract enhanced 144 features EXACTLY like training script.
    all_hands = list of mediapipe landmarks
    """

    # If no hands detected → 144 zeros
    if len(all_hands) == 0:
        return np.zeros(144)

    # -------- HAND 1 --------
    h1 = np.array([[p.x, p.y, p.z] for p in all_hands[0]])
    h1_rel = h1 - h1[0]

    fingertips = [4, 8, 12, 16, 20]
    h1_dist = [np.linalg.norm(h1[i] - h1[0]) for i in fingertips]

    # angles
    h1_vecs = [(h1[i] - h1[0]) for i in fingertips]
    h1_ang = []
    for i in range(len(h1_vecs) - 1):
        cos_val = np.dot(h1_vecs[i], h1_vecs[i+1]) / (
            np.linalg.norm(h1_vecs[i]) * np.linalg.norm(h1_vecs[i+1])
        )
        h1_ang.append(np.arccos(np.clip(cos_val, -1, 1)))

    # -------- HAND 2 or padding --------
    if len(all_hands) > 1:
        h2 = np.array([[p.x, p.y, p.z] for p in all_hands[1]])
        h2_rel = h2 - h2[0]

        h2_dist = [np.linalg.norm(h2[i] - h2[0]) for i in fingertips]

        h2_vecs = [(h2[i] - h2[0]) for i in fingertips]
        h2_ang = []
        for i in range(len(h2_vecs) - 1):
            cos_val = np.dot(h2_vecs[i], h2_vecs[i+1]) / (
                np.linalg.norm(h2_vecs[i]) * np.linalg.norm(h2_vecs[i+1])
            )
            h2_ang.append(np.arccos(np.clip(cos_val, -1, 1)))
    else:
        h2_rel = np.zeros_like(h1_rel)
        h2_dist = np.zeros_like(h1_dist)
        h2_ang = np.zeros_like(h1_ang)

    # -------- CONCAT EXACTLY LIKE TRAINING --------
    return np.concatenate([
        h1_rel.flatten(),
        h2_rel.flatten(),
        np.array(h1_dist),
        np.array(h2_dist),
        np.array(h1_ang),
        np.array(h2_ang)
    ])


# ===== GUI =====
def draw_info_box(image, text, confidence):
    h, w, _ = image.shape
    box_w, box_h = 350, 80
    x, y = 30, 30

    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)

    cv2.putText(image, f"Gesture: {text}", (x + 15, y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(image, f"Confidence: {confidence:.2f}", (x + 15, y + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# ===== MAIN =====
def main():
    cap = cv2.VideoCapture(0)
    last_prediction = None

    print("\nGesture Mode Started (Single + Double Hand)\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        img = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        all_hands = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                all_hands.append(hand_landmarks.landmark)

        # Extract features
        features = extract_features_for_prediction(all_hands).reshape(1, -1)

        # Predict only if any hand detected
        if len(all_hands) > 0:
            proba = model.predict_proba(features)[0]
            idx = np.argmax(proba)
            conf = proba[idx]

            if conf > 0.7:
                pred = label_encoder.inverse_transform([idx])[0]
                draw_info_box(img, pred, conf)

                if pred != last_prediction:
                    speak_in_background(pred)
                    last_prediction = pred
        else:
            last_prediction = None

        cv2.imshow("Gesture Recognition", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
