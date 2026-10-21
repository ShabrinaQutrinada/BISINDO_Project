import os
import numpy as np

for label in sorted(os.listdir("dataset")):
    folder = os.path.join("dataset", label)
    if not os.path.isdir(folder):
        continue

    files = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
    bermasalah = 0

    for f in files:
        data = np.load(os.path.join(folder, f))
        detected = np.sum(np.any(data != 0, axis=1))
        if data.shape != (30, 126) or detected < 27:
            bermasalah += 1

    print(f"{label}: {len(files)} file | bermasalah: {bermasalah}")