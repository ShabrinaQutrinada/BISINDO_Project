import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import torch

from utils import (
    HAND_CONNECTIONS,
    SEQUENCE_LENGTH,
    SignLSTM,
    extract_landmarks,
    normalize_sequence
)


# =========================
# PENGATURAN
# =========================

MODEL_PATH = "models/bisindo_lstm.pt"

CONF_THRESHOLD = 0.70
STABLE_COUNT = 3
MIN_HAND_FRAMES = 27
PREDICT_EVERY = 2
HOLD_SECONDS = 1.5
REPEAT_GAP = 2.0

# Kalau nanti ada kelas LAINNYA,
# kelas tersebut dianggap UNKNOWN
IGNORE_LABELS = {"LAINNYA"}


# =========================
# LOAD MODEL
# =========================

ckpt = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

labels = ckpt["labels"]

model = SignLSTM(
    len(labels)
)

model.load_state_dict(
    ckpt["state_dict"]
)

model.eval()

print("Label model:", labels)


# =========================
# MEDIAPIPE
# =========================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

landmarker = HandLandmarker.create_from_options(
    options
)


# =========================
# WEBCAM
# =========================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not cap.isOpened():
    print("Webcam tidak dapat dibuka.")
    landmarker.close()
    raise SystemExit(1)


# =========================
# VARIABEL
# =========================

buffer = deque(
    maxlen=SEQUENCE_LENGTH
)

recent = deque(
    maxlen=STABLE_COUNT
)

transcript = []

shown_label = ""
shown_until = 0.0

last_added = ""
last_added_time = 0.0

live_text = ""

frame_count = 0
last_ts = -1

start_time = time.time()
prev_time = time.time()

fps = 0.0


# =========================
# FUNGSI
# =========================

def pretty(name):
    return name.replace("_", " ")


def draw_hands(frame, result):

    h, w, _ = frame.shape

    for hand in result.hand_landmarks:

        pts = [
            (
                int(lm.x * w),
                int(lm.y * h)
            )
            for lm in hand
        ]

        for a, b in HAND_CONNECTIONS:

            cv2.line(
                frame,
                pts[a],
                pts[b],
                (255, 255, 255),
                2
            )

        for point in pts:

            cv2.circle(
                frame,
                point,
                4,
                (255, 255, 255),
                -1
            )


# =========================
# PROGRAM UTAMA
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Webcam tidak dapat dibaca.")
        break

    # Sama dengan saat pengambilan dataset
    frame = cv2.flip(
        frame,
        1
    )

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    ts = int(
        (time.time() - start_time)
        * 1000
    )

    if ts <= last_ts:
        ts = last_ts + 1

    last_ts = ts

    result = landmarker.detect_for_video(
        mp_image,
        ts
    )

    buffer.append(
        extract_landmarks(result)
    )

    frame_count += 1

    draw_hands(
        frame,
        result
    )


    # =========================
    # PREDIKSI
    # =========================

    if (
        len(buffer) == SEQUENCE_LENGTH
        and frame_count % PREDICT_EVERY == 0
    ):

        seq = np.array(
            buffer,
            dtype=np.float32
        )

        hand_frames = int(
            np.sum(
                np.any(
                    seq != 0,
                    axis=1
                )
            )
        )

        if hand_frames >= MIN_HAND_FRAMES:

            normalized = normalize_sequence(
                seq
            )

            x = torch.from_numpy(
                normalized
            ).unsqueeze(0)

            with torch.no_grad():

                output = model(x)

                probs = torch.softmax(
                    output,
                    dim=1
                )[0]

            conf, idx = torch.max(
                probs,
                dim=0
            )

            conf = conf.item()
            idx = idx.item()

            predicted_label = labels[idx]

            is_ignored = (
                predicted_label
                in IGNORE_LABELS
            )


            # =========================
            # UNKNOWN / VALID
            # =========================

            if (
                conf >= CONF_THRESHOLD
                and not is_ignored
            ):

                live_text = (
                    f"{pretty(predicted_label)} "
                    f"({conf * 100:.0f}%)"
                )

                recent.append(idx)

            else:

                live_text = (
                    f"UNKNOWN "
                    f"({conf * 100:.0f}%)"
                )

                recent.append(-1)


            # =========================
            # CEK STABIL
            # =========================

            if (
                len(recent) == STABLE_COUNT
                and len(set(recent)) == 1
                and recent[0] != -1
            ):

                word = pretty(
                    labels[recent[0]]
                )

                now = time.time()

                shown_label = word

                shown_until = (
                    now + HOLD_SECONDS
                )

                if (
                    word != last_added
                    or
                    now - last_added_time
                    > REPEAT_GAP
                ):

                    transcript.append(
                        word
                    )

                    last_added = word

                last_added_time = now

        else:

            recent.append(-1)
            live_text = ""


    # =========================
    # FPS
    # =========================

    now = time.time()

    fps = (
        0.9 * fps
        +
        0.1
        * (
            1.0
            /
            max(
                now - prev_time,
                1e-6
            )
        )
    )

    prev_time = now


    # =========================
    # TAMPILAN
    # =========================

    h, w, _ = frame.shape

    cv2.rectangle(
        frame,
        (0, h - 70),
        (w, h),
        (30, 30, 30),
        -1
    )


    # Kata yang berhasil dikenali
    if now < shown_until:

        cv2.putText(
            frame,
            shown_label,
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            4
        )

    else:

        cv2.putText(
            frame,
            "Menunggu isyarat...",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )


    # Prediksi langsung
    if live_text:

        cv2.putText(
            frame,
            f"Tebakan: {live_text}",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )


    # Transcript
    text = " ".join(
        transcript[-5:]
    )

    cv2.putText(
        frame,
        text,
        (20, h - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )


    # FPS
    cv2.putText(
        frame,
        f"FPS {fps:.0f}",
        (w - 110, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )


    cv2.imshow(
        "BISINDO - Sign Language to Text",
        frame
    )


    # =========================
    # KEYBOARD
    # =========================

    key = cv2.waitKey(1) & 0xFF

    # Q = keluar
    if key == ord("q"):
        break

    # C = hapus transcript
    if key == ord("c"):
        transcript.clear()


# =========================
# SELESAI
# =========================

cap.release()
landmarker.close()
cv2.destroyAllWindows()