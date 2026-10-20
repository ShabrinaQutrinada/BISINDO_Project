import cv2
import mediapipe as mp
import time

# Hubungan antar 21 titik landmark tangan
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# =========================
# SETUP MEDIAPIPE
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
    min_tracking_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)

# =========================
# BUKA WEBCAM
# =========================

cap = cv2.VideoCapture(0)

start_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Webcam tidak dapat dibaca.")
        break

    # Membalik kamera seperti cermin
    frame = cv2.flip(frame, 1)

    # OpenCV BGR -> RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Ubah frame menjadi format MediaPipe
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Timestamp untuk mode video
    timestamp_ms = int(
        (time.time() - start_time) * 1000
    )

    # Deteksi tangan
    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    # =========================
    # GAMBAR LANDMARK TANGAN
    # =========================

    if result.hand_landmarks:
        height, width, _ = frame.shape

        for hand_landmarks in result.hand_landmarks:
            points = []

            # Ambil koordinat 21 titik
            for landmark in hand_landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                points.append((x, y))

                # Gambar titik
                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (255, 255, 255),
                    -1
                )

            # Gambar garis penghubung
            for start, end in HAND_CONNECTIONS:
                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (255, 255, 255),
                    2
                )

    # Tampilkan hasil
    cv2.imshow(
        "BISINDO - Hand Landmark Detection",
        frame
    )

    # Tekan q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()