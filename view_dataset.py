import os
import sys
import cv2
import numpy as np
from utils import HAND_CONNECTIONS

if len(sys.argv) < 2:
    print("Contoh: python view_dataset.py HALO 3")
    sys.exit()

label = sys.argv[1].upper()
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 3

folder = os.path.join("dataset", label)
files = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))[:limit]
W, H = 640, 480
FONT = cv2.FONT_HERSHEY_SIMPLEX

for file in files:
    data = np.load(os.path.join(folder, file))
    for n, frame_data in enumerate(data):
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        for offset in (0, 63):
            pts = []
            for i in range(21):
                x = frame_data[offset + i * 3]
                y = frame_data[offset + i * 3 + 1]
                if x == 0 and y == 0:
                    pts.append(None)
                else:
                    p = (int(x * W), int(y * H))
                    pts.append(p)
                    cv2.circle(canvas, p, 5, (255, 255, 255), -1)
            for a, b in HAND_CONNECTIONS:
                if pts[a] and pts[b]:
                    cv2.line(canvas, pts[a], pts[b], (255, 255, 255), 2)

        cv2.putText(canvas, f"{label} - {file}", (20, 40), FONT, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"Frame {n + 1}/{len(data)}", (20, 80), FONT, 0.7, (255, 255, 255), 2)
        cv2.imshow("View BISINDO Dataset", canvas)

        if cv2.waitKey(100) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            sys.exit()

cv2.destroyAllWindows()