
import numpy as np
import torch.nn as nn

SEQUENCE_LENGTH = 30
RAW_DIM = 126       # 2 tangan x 21 landmark x (x, y, z)
FEATURE_DIM = 130   # 2 tangan x (63 bentuk tangan + 2 posisi pergelangan)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def extract_landmarks(result):
    """Hasil MediaPipe -> vektor 126 nilai (Left: 0-62, Right: 63-125)."""
    data = np.zeros(RAW_DIM, dtype=np.float32)

    if not result.hand_landmarks:
        return data

    for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        offset = 0 if handedness[0].category_name == "Left" else 63
        for i, lm in enumerate(hand_landmarks):
            idx = offset + i * 3
            data[idx] = lm.x
            data[idx + 1] = lm.y
            data[idx + 2] = lm.z

    return data


def normalize_sequence(seq):
    """
    (T, 126) mentah -> (T, 130) ternormalisasi.

    Per tangan:
    - bentuk tangan: landmark dikurangi pergelangan lalu dibagi ukuran tangan
      (jarak pergelangan ke pangkal jari tengah), sehingga tidak tergantung
      jarak ke kamera dan posisi di layar.
    - posisi pergelangan (x, y) tetap disimpan agar gerakan tangan
      (misal melambai) masih terbaca.
    Tangan yang tidak terdeteksi tetap bernilai 0.
    """
    seq = np.asarray(seq, dtype=np.float32)
    out = np.zeros((seq.shape[0], FEATURE_DIM), dtype=np.float32)

    for t in range(seq.shape[0]):
        for h, offset in enumerate((0, 63)):
            hand = seq[t, offset:offset + 63]
            if not np.any(hand):
                continue

            pts = hand.reshape(21, 3)
            wrist = pts[0].copy()
            rel = pts - wrist

            scale = float(np.linalg.norm(rel[9, :2]))
            if scale < 1e-6:
                scale = 1.0
            rel = rel / scale

            base = h * 65
            out[t, base:base + 63] = rel.reshape(-1)
            out[t, base + 63:base + 65] = wrist[:2]

    return out


class SignLSTM(nn.Module):
    def __init__(self, num_classes, input_dim=FEATURE_DIM, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden, num_layers=2, batch_first=True, dropout=0.3
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, num_classes))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])
