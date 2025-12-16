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
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils


def extract_hand_landmarks(hand_landmarks):
    """Extract 21 landmark coordinates as flat list"""
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return coords


def collect_gesture_samples(sign_label, samples_needed=50):
    print(f"\nCollecting samples for gesture: {sign_label}")
    print("Show the gesture with 1 or 2 hands.")
    print("Press 'q' to stop early.\n")

    cap = cv2.VideoCapture(0)
    all_rows = []
    collected = 0
    last_time = time.time()

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

        hand_coords = []

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_lm, mp_hands.HAND_CONNECTIONS)
                hand_coords.append(extract_hand_landmarks(hand_lm))

        # --- Auto sample save every 0.3 second ---
        if time.time() - last_time > 0.3 and len(hand_coords) > 0:

            # If only 1 hand detected → pad zeros for 2nd hand
            if len(hand_coords) == 1:
                hand1 = hand_coords[0]
                hand2 = [0] * 63   # 21 points → 63 values
            else:
                # Two hands detected → both stored
                hand1 = hand_coords[0]
                hand2 = hand_coords[1]

            row = hand1 + hand2 + [sign_label]
            all_rows.append(row)
            collected += 1
            last_time = time.time()

            print(f"Saved sample {collected}/{samples_needed}")

        cv2.imshow("Gesture Data Collection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return all_rows


def save_to_csv(data, filename="double_hand_dataset.csv"):
    cols = []

    # HAND 1 columns
    for i in range(21):
        cols += [f"h1_x{i}", f"h1_y{i}", f"h1_z{i}"]

    # HAND 2 columns
    for i in range(21):
        cols += [f"h2_x{i}", f"h2_y{i}", f"h2_z{i}"]

    cols.append("label")

    df = pd.DataFrame(data, columns=cols)

    if os.path.exists(filename):
        old = pd.read_csv(filename)
        df = pd.concat([old, df], ignore_index=True)

    df.to_csv(filename, index=False)
    print(f"Saved {len(data)} new samples to: {filename}")


def main():
    while True:
        label = input("\nEnter gesture name (or 'quit'): ").strip()
        if label.lower() == "quit":
            break

        data = collect_gesture_samples(label, samples_needed=50)
        save_to_csv(data)

    print("\nDataset collection finished!")


if __name__ == "__main__":
    main()
