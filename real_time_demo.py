# import cv2
# import mediapipe as mp
# import numpy as np
# import joblib
# import pyttsx3
# import threading

# # load
# model = joblib.load("gesture_recognition_model.pkl")
# label_encoder = joblib.load("label_encoder.pkl")

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=2,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.7
# )
# mp_drawing = mp.solutions.drawing_utils

# # ---------------- Voice control state ----------------
# VOICE_ENABLED = True

# # button area (x1,y1,x2,y2)
# VOICE_BTN = (450, 20, 620, 70)

# def speak_text(text, rate=155, volume=1.0):
#     engine = pyttsx3.init()
#     engine.setProperty('rate', rate)
#     engine.setProperty('volume', volume)
#     engine.say(text)
#     engine.runAndWait()

# def speak_in_background(text):
#     if not VOICE_ENABLED:
#         return
#     t = threading.Thread(target=lambda: speak_text(text))
#     t.daemon = True
#     t.start()

# # ---------------- Feature extraction ----------------
# def extract_for_predict(all_hands):
#     features_list = []

#     for lm in all_hands:
#         pts = np.array([[p.x, p.y, p.z] for p in lm.landmark])

#         wrist = pts[0]
#         rel = (pts - wrist).flatten()
#         dists = [np.linalg.norm(pts[i] - wrist) for i in [4,8,12,16,20]]

#         vecs = [pts[i] - wrist for i in [4,8,12,16,20]]
#         angs = []
#         for i in range(len(vecs)-1):
#             cos_val = np.dot(vecs[i], vecs[i+1]) / (
#                 np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[i+1]) + 1e-9
#             )
#             angs.append(np.arccos(np.clip(cos_val, -1, 1)))

#         features_list.append(np.concatenate([rel, dists, angs]))

#     if len(features_list) == 0:
#         return np.zeros(144)
#     elif len(features_list) == 1:
#         return np.concatenate([features_list[0], np.zeros_like(features_list[0])])
#     else:
#         return np.concatenate([features_list[0], features_list[1]])

# # ---------------- UI helpers ----------------
# def draw_prediction_box(img, text, conf):
#     cv2.rectangle(img, (20,20), (420,110), (0,0,0), -1)
#     cv2.putText(img, f"Gesture: {text}", (30,60),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
#     cv2.putText(img, f"Conf: {conf:.2f}", (30,95),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

# def draw_voice_button(img):
#     global VOICE_ENABLED
#     x1,y1,x2,y2 = VOICE_BTN
#     color = (0,200,0) if VOICE_ENABLED else (0,0,200)
#     cv2.rectangle(img, (x1,y1), (x2,y2), color, -1)
#     text = "VOICE: ON" if VOICE_ENABLED else "VOICE: OFF"
#     cv2.putText(img, text, (x1+15, y1+35),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

# def mouse_callback(event, x, y, flags, param):
#     global VOICE_ENABLED
#     if event == cv2.EVENT_LBUTTONDOWN:
#         x1,y1,x2,y2 = VOICE_BTN
#         if x1 <= x <= x2 and y1 <= y <= y2:
#             VOICE_ENABLED = not VOICE_ENABLED
#             print("Voice:", "ON" if VOICE_ENABLED else "OFF")

# # ---------------- Main loop ----------------
# def main():
#     cap = cv2.VideoCapture(0)
#     last_pred = None

#     cv2.namedWindow("Gesture Recognition (Live)")
#     cv2.setMouseCallback("Gesture Recognition (Live)", mouse_callback)

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             continue

#         img = cv2.cvtColor(cv2.flip(frame,1), cv2.COLOR_BGR2RGB)
#         results = hands.process(img)
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#         all_hands = []
#         if results.multi_hand_landmarks:
#             left, right = None, None
#             for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
#                 label = results.multi_handedness[idx].classification[0].label
#                 if label == "Left":
#                     left = hand_landmarks
#                 else:
#                     right = hand_landmarks

#             if left is not None:
#                 all_hands.append(left)
#                 mp_drawing.draw_landmarks(img, left, mp_hands.HAND_CONNECTIONS)
#             if right is not None:
#                 all_hands.append(right)
#                 mp_drawing.draw_landmarks(img, right, mp_hands.HAND_CONNECTIONS)

#         feats = extract_for_predict(all_hands).reshape(1, -1)

#         if len(all_hands) > 0:
#             proba = model.predict_proba(feats)[0]
#             idx = np.argmax(proba)
#             conf = proba[idx]

#             if conf > 0.65:
#                 pred = label_encoder.inverse_transform([idx])[0]
#                 draw_prediction_box(img, pred, conf)

#                 if pred != last_pred:
#                     speak_in_background(pred)
#                     last_pred = pred
#         else:
#             last_pred = None

#         draw_voice_button(img)

#         cv2.imshow("Gesture Recognition (Live)", img)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()
import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
import threading
import time

# ==================== CONFIGURATION ====================
CONFIDENCE_THRESHOLD = 0.40  # Adjust based on your needs (0.3-0.6)
STABILITY_TIME = 0.3  # Seconds before confirming gesture change
MIN_DETECTION_CONF = 0.5
MIN_TRACKING_CONF = 0.5

# ==================== LOAD MODEL ====================
try:
    model = joblib.load("gesture_recognition_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    print(f"✓ Model loaded successfully")
    print(f"✓ Gestures available: {list(label_encoder.classes_)}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit(1)

# ==================== MEDIAPIPE SETUP ====================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=MIN_DETECTION_CONF,
    min_tracking_confidence=MIN_TRACKING_CONF
)
mp_drawing = mp.solutions.drawing_utils

# ==================== VOICE SYSTEM ====================
VOICE_ENABLED = True
VOICE_BTN = (450, 20, 620, 70)
speech_lock = threading.Lock()
is_speaking = False

def speak_text(text, rate=150, volume=1.0):
    """Text-to-speech function"""
    global is_speaking
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', rate)
        engine.setProperty('volume', volume)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Speech error: {e}")
    finally:
        with speech_lock:
            is_speaking = False

def speak_in_background(text):
    """Non-blocking speech"""
    global is_speaking
    if not VOICE_ENABLED:
        return
    
    with speech_lock:
        if is_speaking:
            return
        is_speaking = True
    
    thread = threading.Thread(target=lambda: speak_text(text), daemon=True)
    thread.start()

# ==================== FEATURE EXTRACTION ====================
def extract_features(hand_landmarks_list):
    """
    Extract features from detected hands
    Returns 144-dimensional feature vector (72 per hand)
    """
    features_list = []

    for hand_landmarks in hand_landmarks_list:
        # Get all 21 landmarks as numpy array
        points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        
        # Use wrist (landmark 0) as reference point
        wrist = points[0]
        
        # 1. Relative positions (63 features: 21 landmarks × 3 coords)
        relative_positions = (points - wrist).flatten()
        
        # 2. Fingertip distances to wrist (5 features)
        fingertip_indices = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
        distances = [np.linalg.norm(points[i] - wrist) for i in fingertip_indices]
        
        # 3. Inter-finger angles (4 features)
        finger_vectors = [points[i] - wrist for i in fingertip_indices]
        angles = []
        for i in range(len(finger_vectors) - 1):
            v1, v2 = finger_vectors[i], finger_vectors[i + 1]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angles.append(angle)
        
        # Combine all features (72 per hand)
        hand_features = np.concatenate([relative_positions, distances, angles])
        features_list.append(hand_features)
    
    # Handle one hand or two hands
    if len(features_list) == 0:
        return np.zeros(144)
    elif len(features_list) == 1:
        # Pad with zeros for second hand
        return np.concatenate([features_list[0], np.zeros(72)])
    else:
        # Both hands present
        return np.concatenate([features_list[0], features_list[1]])

# ==================== UI DRAWING FUNCTIONS ====================
def draw_prediction_panel(frame, gesture, confidence, all_probs, all_classes):
    """Draw main prediction panel with top predictions"""
    # Background panel
    cv2.rectangle(frame, (10, 10), (430, 180), (40, 40, 40), -1)
    cv2.rectangle(frame, (10, 10), (430, 180), (0, 255, 0), 2)
    
    # Main gesture (large)
    cv2.putText(frame, f"{gesture}", (20, 55),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)
    
    # Confidence bar
    bar_width = int(400 * confidence)
    cv2.rectangle(frame, (20, 70), (420, 90), (60, 60, 60), -1)
    cv2.rectangle(frame, (20, 70), (20 + bar_width, 90), (0, 255, 0), -1)
    cv2.putText(frame, f"{confidence:.1%}", (340, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Top 3 predictions
    cv2.putText(frame, "Top Predictions:", (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    top_indices = np.argsort(all_probs)[-3:][::-1]
    y_pos = 135
    for rank, idx in enumerate(top_indices, 1):
        prob_pct = all_probs[idx] * 100
        text = f"{rank}. {all_classes[idx]}: {prob_pct:.1f}%"
        color = (0, 255, 0) if rank == 1 else (180, 180, 180)
        cv2.putText(frame, text, (30, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_pos += 20

def draw_voice_toggle(frame):
    """Draw voice on/off button"""
    x1, y1, x2, y2 = VOICE_BTN
    color = (0, 200, 0) if VOICE_ENABLED else (0, 0, 200)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    
    status = "ON" if VOICE_ENABLED else "OFF"
    cv2.putText(frame, f"Voice: {status}", (x1 + 20, y1 + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def draw_hand_info(frame, num_hands, hand_types):
    """Show hand detection info"""
    y_pos = frame.shape[0] - 20
    
    if num_hands == 0:
        text = "No hands detected"
        color = (0, 0, 255)
    else:
        text = f"Hands: {num_hands} ({', '.join(hand_types)})"
        color = (0, 255, 255)
    
    cv2.putText(frame, text, (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def draw_instructions(frame):
    """Draw usage instructions"""
    instructions = [
        "Press 'Q' to quit",
        "Click button to toggle voice"
    ]
    
    y_start = frame.shape[0] - 70
    for i, text in enumerate(instructions):
        cv2.putText(frame, text, (frame.shape[1] - 300, y_start + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

# ==================== MOUSE CALLBACK ====================
def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks on voice button"""
    global VOICE_ENABLED
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1, x2, y2 = VOICE_BTN
        if x1 <= x <= x2 and y1 <= y <= y2:
            VOICE_ENABLED = not VOICE_ENABLED
            print(f"Voice: {'ON' if VOICE_ENABLED else 'OFF'}")

# ==================== MAIN LOOP ====================
def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Error: Cannot open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Window setup
    window_name = "Gesture Recognition System"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # State variables
    last_gesture = None
    last_change_time = time.time()
    
    print("\n" + "="*50)
    print("GESTURE RECOGNITION SYSTEM STARTED")
    print("="*50)
    print(f"Available gestures: {list(label_encoder.classes_)}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print("Press 'Q' to quit\n")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Flip and convert frame
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hands
        results = hands.process(rgb_frame)
        
        # Collect hand landmarks
        hand_landmarks_list = []
        hand_types = []
        
        if results.multi_hand_landmarks:
            # Sort hands (left first, then right)
            left_hand, right_hand = None, None
            
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness = results.multi_handedness[idx].classification[0].label
                
                if handedness == "Left":
                    left_hand = hand_landmarks
                else:
                    right_hand = hand_landmarks
            
            # Add in consistent order
            if left_hand:
                hand_landmarks_list.append(left_hand)
                hand_types.append("Left")
                mp_drawing.draw_landmarks(frame, left_hand, mp_hands.HAND_CONNECTIONS)
            
            if right_hand:
                hand_landmarks_list.append(right_hand)
                hand_types.append("Right")
                mp_drawing.draw_landmarks(frame, right_hand, mp_hands.HAND_CONNECTIONS)
        
        # Predict gesture
        if len(hand_landmarks_list) > 0:
            features = extract_features(hand_landmarks_list).reshape(1, -1)
            
            probabilities = model.predict_proba(features)[0]
            predicted_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_idx]
            predicted_gesture = label_encoder.inverse_transform([predicted_idx])[0]
            
            # Show prediction if confidence is high enough
            if confidence >= CONFIDENCE_THRESHOLD:
                draw_prediction_panel(frame, predicted_gesture, confidence,
                                    probabilities, label_encoder.classes_)
                
                # Announce gesture change (with stability check)
                if predicted_gesture != last_gesture:
                    if time.time() - last_change_time >= STABILITY_TIME:
                        speak_in_background(predicted_gesture)
                        last_gesture = predicted_gesture
                        print(f"→ Detected: {predicted_gesture} ({confidence:.1%})")
                        last_change_time = time.time()
                else:
                    last_change_time = time.time()
        else:
            last_gesture = None
        
        # Draw UI elements
        draw_voice_toggle(frame)
        draw_hand_info(frame, len(hand_landmarks_list), hand_types)
        draw_instructions(frame)
        
        # Show frame
        cv2.imshow(window_name, frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\n" + "="*50)
    print("GESTURE RECOGNITION SYSTEM STOPPED")
    print("="*50)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n✓ Interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()