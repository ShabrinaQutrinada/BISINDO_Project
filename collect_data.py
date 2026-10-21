
import argparse
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

from utils import SEQUENCE_LENGTH, RAW_DIM, extract_landmarks

parser = argparse.ArgumentParser()
parser.add_argument("labels", nargs="+", help="nama kata, contoh: HALO TOLONG")
parser.add_argument("--n", type=int, default=40, help="jumlah sequence BARU per kata")
parser.add_argument("--person", default="sequence", help="nama perekam")
parser.add_argument("--min-detected", type=int, default=27,
                    help="minimal frame yang tangannya terdeteksi")
args = parser.parse_args()

# =========================
# SETUP MEDIAPIPE
# =========================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = HandLandmarker.create_from_options(options)

# =========================
# WEBCAM
# =========================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Webcam tidak dapat dibuka.")
    landmarker.close()
    sys.exit(1)

WINDOW = "Collect BISINDO Dataset"
start_time = time.time()
last_ts = -1


def put(frame, text, y, scale=0.8):
    cv2.putText(frame, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 2)


def stop():
    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    sys.exit(0)


def read_frame():
    ret, frame = cap.read()
    if not ret:
        print("Webcam gagal membaca frame.")
        stop()
    return cv2.flip(frame, 1)  # mirror, harus sama dengan main.py


def check_quit():
    if cv2.waitKey(1) & 0xFF == ord("q"):
        stop()


def next_index(folder, person):
    nums = []
    for f in os.listdir(folder):
        if f.startswith(person + "_") and f.endswith(".npy"):
            try:
                nums.append(int(f[len(person) + 1:-4]))
            except ValueError:
                pass
    return max(nums, default=0) + 1


# warm-up kamera
time.sleep(1)
for _ in range(10):
    cap.read()

# =========================
# PENGAMBILAN DATASET
# =========================

for label in args.labels:
    label = label.upper()
    folder = os.path.join("dataset", label)
    os.makedirs(folder, exist_ok=True)

    idx = next_index(folder, args.person)
    saved = 0

    while saved < args.n:
        # countdown 2 detik
        t0 = time.time()
        while time.time() - t0 < 2:
            frame = read_frame()
            put(frame, f"Siap: {label}", 50, 1)
            put(frame, f"Tersimpan {saved}/{args.n}", 90)
            put(frame, f"Mulai dalam {2 - int(time.time() - t0)}", 125)
            cv2.imshow(WINDOW, frame)
            check_quit()

        # rekam 30 frame
        seq = []
        detected = 0
        for fn in range(SEQUENCE_LENGTH):
            frame = read_frame()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            ts = int((time.time() - start_time) * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts

            result = landmarker.detect_for_video(mp_image, ts)
            seq.append(extract_landmarks(result))

            if result.hand_landmarks:
                detected += 1

            put(frame, f"RECORDING: {label}", 50, 1)
            put(frame, f"Frame {fn + 1}/{SEQUENCE_LENGTH}", 90)
            put(frame, "HAND DETECTED" if result.hand_landmarks else "NO HAND", 125, 0.7)
            cv2.imshow(WINDOW, frame)
            check_quit()

        seq = np.array(seq, dtype=np.float32)

        if seq.shape != (SEQUENCE_LENGTH, RAW_DIM) or detected < args.min_detected:
            print(f"[ULANG] {label}: tangan hanya terdeteksi {detected}/{SEQUENCE_LENGTH} frame")
            continue

        path = os.path.join(folder, f"{args.person}_{idx:03d}.npy")
        np.save(path, seq)
        print(f"Tersimpan: {path} | terdeteksi {detected}/{SEQUENCE_LENGTH}")
        idx += 1
        saved += 1

cap.release()
landmarker.close()
cv2.destroyAllWindows()
print("Pengambilan dataset selesai.")
