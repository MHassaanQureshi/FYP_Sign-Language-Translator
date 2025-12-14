# collect_validated.py
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import time
import os

# ------- Setup -------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)
mp_drawing = mp.solutions.drawing_utils

# state for stability per hand (uses handedness label: "Left"/"Right")
stability_state = {
    "Left": {"last_pts": None, "stable_count": 0, "last_saved": 0},
    "Right": {"last_pts": None, "stable_count": 0, "last_saved": 0}
}

def extract_hand_landmarks(hand_landmarks):
    """Extract 21 landmark coordinates as flat list (x,y,z)"""
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return coords

def is_hand_complete(landmarks):
    """Check landmarks not too close to edges (visible & complete)"""
    for lm in landmarks.landmark:
        if lm.x < 0.02 or lm.x > 0.98 or lm.y < 0.02 or lm.y > 0.98:
            return False
    return True

def is_centered(landmarks):
    pts = np.array([[lm.x, lm.y] for lm in landmarks.landmark])
    avg_x = np.mean(pts[:, 0])
    avg_y = np.mean(pts[:, 1])
    return 0.25 < avg_x < 0.75 and 0.2 < avg_y < 0.8

def is_angle_correct(landmarks):
    return True
    # # use wrist (0) and index finger tip (8) vector to check tilt
    # wrist = np.array([landmarks.landmark[0].x, landmarks.landmark[0].y, landmarks.landmark[0].z])
    # index = np.array([landmarks.landmark[8].x, landmarks.landmark[8].y, landmarks.landmark[8].z])
    # vec = index - wrist
    # # vertical component small => roughly facing camera (tweak threshold if needed)
    # vertical = abs(vec[1])
    # return vertical < 0.12

def is_stable(hand_label, landmarks, movement_threshold=0.015, required_frames=6):
    """
    Track small movements across frames. Required_frames * frame_interval ~ 0.2-0.3s
    movement_threshold controls sensitivity
    """
    state = stability_state[hand_label]
    pts = np.array([[lm.x, lm.y] for lm in landmarks.landmark]).flatten()

    if state["last_pts"] is None:
        state["last_pts"] = pts
        state["stable_count"] = 0
        return False

    movement = np.linalg.norm(pts - state["last_pts"])

    if movement < movement_threshold:
        state["stable_count"] += 1
    else:
        state["stable_count"] = 0

    state["last_pts"] = pts
    return state["stable_count"] >= required_frames

def get_handedness_label(results, idx):
    """Return 'Left' or 'Right' for the given hand index using multi_handedness"""
    try:
        return results.multi_handedness[idx].classification[0].label
    except Exception:
        return None

def collect_gesture_samples(sign_label, samples_needed=50, save_filename="double_hand_dataset.csv"):
    print(f"\nCollecting samples for gesture: {sign_label}")
    print("Show the gesture with 1 or 2 hands in center, keep stable for ~0.25s.")
    print("Press 'q' to stop early.\n")

    cap = cv2.VideoCapture(0)
    all_rows = []
    collected = 0
    last_capture_time = 0
    min_capture_interval = 0.5  # seconds between saved samples

    while cap.isOpened() and collected < samples_needed:
        ret, frame = cap.read()
        if not ret:
            continue

        img = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.putText(img, f"Gesture: {sign_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"Samples: {collected}/{samples_needed}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # prepared storage: ensure left/right mapping always
        left_coords = None
        right_coords = None

        if results.multi_hand_landmarks and results.multi_handedness:
            # iterate through detected hands and use handedness label to place left/right
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = get_handedness_label(results, idx)  # "Left" or "Right"
                if label is None:
                    continue

                # draw landmarks
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # validation pipeline per hand
                complete = is_hand_complete(hand_landmarks)
                centered = is_centered(hand_landmarks)
                angle_ok = is_angle_correct(hand_landmarks)
                # stable = is_stable(label, hand_landmarks)
                stable = is_stable(
                    label,
                    hand_landmarks,
                    movement_threshold=0.03,
                    required_frames=3
                )


                # show status on frame for debugging
                status = f"{label}: Cmp:{int(complete)} Ctr:{int(centered)} Ang:{int(angle_ok)} Stb:{int(stable)}"
                cv2.putText(img, status, (10, 110 + (20 * idx)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # if hand passes all checks, extract coords, else ignore
                if complete and centered and angle_ok and stable:
                    coords = extract_hand_landmarks(hand_landmarks)
                    if label == "Left":
                        left_coords = coords
                    else:
                        right_coords = coords

        # Only save if capture conditions met:
        # - Single-hand: that hand must pass validation and time since last save > min_capture_interval
        # - Double-hand: both left and right must be non-None and time condition satisfied
        now = time.time()
        can_save = False
        if left_coords is not None or right_coords is not None:
            # Determine if both required for this sign: we will accept single or double depending on shown hands,
            # but require the hand(s) that exist to have passed validation.
            if now - last_capture_time > min_capture_interval:
                # create row: left then right; if missing -> zeros
                if left_coords is None:
                    left_coords = [0] * 63
                if right_coords is None:
                    right_coords = [0] * 63

                row = left_coords + right_coords + [sign_label]
                all_rows.append(row)
                collected += 1
                last_capture_time = now
                print(f"Saved sample {collected}/{samples_needed}")

        cv2.imshow("Gesture Data Collection (Validated)", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # save to csv
    cols = []
    for i in range(21):
        cols += [f"h1_x{i}", f"h1_y{i}", f"h1_z{i}"]
    for i in range(21):
        cols += [f"h2_x{i}", f"h2_y{i}", f"h2_z{i}"]
    cols.append("label")

    df = pd.DataFrame(all_rows, columns=cols)
    if os.path.exists(save_filename):
        old = pd.read_csv(save_filename)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(save_filename, index=False)
    print(f"\nSaved {len(all_rows)} samples to {save_filename}")
    return save_filename

def main():
    while True:
        label = input("\nEnter gesture name (or 'quit'): ").strip()
        if label.lower() == "quit":
            break
        try:
            n = int(input("How many samples to collect? (default 50): ") or "50")
        except:
            n = 50
        collect_gesture_samples(label, samples_needed=n)

if __name__ == "__main__":
    main()
