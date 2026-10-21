import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from utils import RAW_DIM, SEQUENCE_LENGTH, SignLSTM, normalize_sequence


# =========================
# PENGATURAN
# =========================

DATASET_DIR = "dataset"
MODEL_PATH = "models/bisindo_lstm.pt"
RESULT_DIR = "results"

EPOCHS = 150
BATCH_SIZE = 16
LEARNING_RATE = 0.001
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# =========================
# LOAD DATASET
# =========================

def load_dataset():
    labels = sorted([
        name for name in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, name))
    ])

    X = []
    y = []

    for class_id, label in enumerate(labels):
        folder = os.path.join(DATASET_DIR, label)

        for file in sorted(os.listdir(folder)):
            if not file.endswith(".npy"):
                continue

            data = np.load(os.path.join(folder, file))

            if data.shape != (SEQUENCE_LENGTH, RAW_DIM):
                print(f"Lewati {label}/{file}: shape salah")
                continue

            X.append(normalize_sequence(data))
            y.append(class_id)

    return (
        np.array(X, dtype=np.float32),
        np.array(y),
        labels
    )


# =========================
# AUGMENTATION
# =========================

def augment(x):
    scale = torch.empty(
        x.shape[0], 1, 1
    ).uniform_(0.95, 1.05)

    noise = (
        0.01
        * torch.randn_like(x)
        * (x != 0)
    )

    x = x * scale + noise

    # Geser waktu: meniru isyarat yang terpotong saat live
    out = torch.zeros_like(x)
    for i in range(x.shape[0]):
        s = int(torch.randint(-8, 9, (1,)))
        if s > 0:
            out[i, s:] = x[i, :-s]
        elif s < 0:
            out[i, :s] = x[i, -s:]
        else:
            out[i] = x[i]

    return out


# =========================
# TRAINING
# =========================

def main():

    X, y, labels = load_dataset()

    print("Jumlah sequence per kata:")

    for i, label in enumerate(labels):
        jumlah = np.sum(y == i)
        print(f"  {label}: {jumlah}")

    # Minimal harus ada 2 label
    if len(labels) < 2:
        print("Training membutuhkan minimal 2 kelas/label.")
        return

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=SEED
    )

    print(
        f"Train: {len(X_train)} | "
        f"Validasi: {len(X_val)}"
    )

    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train).long()

    X_val = torch.from_numpy(X_val)
    y_val_tensor = torch.from_numpy(y_val).long()

    # Model
    model = SignLSTM(len(labels))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    loss_function = nn.CrossEntropyLoss()

    best_accuracy = 0
    best_val_loss = float("inf")
    best_state = None

    train_losses = []
    val_losses = []
    val_accuracies = []

    # =========================
    # LOOP TRAINING
    # =========================

    for epoch in range(1, EPOCHS + 1):

        model.train()

        permutation = torch.randperm(
            len(X_train)
        )

        total_loss = 0

        for i in range(
            0,
            len(X_train),
            BATCH_SIZE
        ):

            index = permutation[
                i:i + BATCH_SIZE
            ]

            xb = augment(
                X_train[index]
            )

            yb = y_train[index]

            optimizer.zero_grad()

            output = model(xb)

            loss = loss_function(
                output,
                yb
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validasi
        model.eval()

        with torch.no_grad():

            val_output = model(X_val)

            val_loss = loss_function(
                val_output,
                y_val_tensor
            ).item()

            prediction = val_output.argmax(1)

            val_accuracy = (
                prediction == y_val_tensor
            ).float().mean().item()

        train_losses.append(total_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Simpan model terbaik:
        # akurasi lebih tinggi, ATAU akurasi sama tapi loss lebih kecil
        if val_accuracy > best_accuracy or (
            val_accuracy == best_accuracy and val_loss < best_val_loss
        ):

            best_accuracy = val_accuracy
            best_val_loss = val_loss

            best_state = {
                key: value.clone()
                for key, value
                in model.state_dict().items()
            }

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d} | "
                f"val_loss {val_loss:.4f} | "
                f"val_acc {val_accuracy:.3f}"
            )

    # =========================
    # SIMPAN MODEL
    # =========================

    os.makedirs("models", exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    torch.save(
        {
            "state_dict": best_state,
            "labels": labels
        },
        MODEL_PATH
    )

    print(
        f"\nModel terbaik "
        f"(val_acc {best_accuracy:.3f}, "
        f"val_loss {best_val_loss:.4f}) "
        f"disimpan di {MODEL_PATH}"
    )

    # =========================
    # EVALUASI
    # =========================

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        pred = model(
            X_val
        ).argmax(1).numpy()

    print(
        "\n=== Laporan klasifikasi ==="
    )

    print(
        classification_report(
            y_val,
            pred,
            target_names=labels,
            zero_division=0
        )
    )

    # Confusion Matrix
    cm = confusion_matrix(
        y_val,
        pred
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)

    plt.xticks(
        range(len(labels)),
        labels,
        rotation=45
    )

    plt.yticks(
        range(len(labels)),
        labels
    )

    plt.xlabel("Prediksi")
    plt.ylabel("Sebenarnya")
    plt.title("Confusion Matrix")

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            "confusion_matrix.png"
        )
    )

    plt.close()

    # Training Curve
    plt.figure(figsize=(7, 4))

    plt.plot(
        val_accuracies,
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Curve")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            "training_curves.png"
        )
    )

    plt.close()

    print(
        f"Hasil grafik disimpan di "
        f"{RESULT_DIR}/"
    )


if __name__ == "__main__":
    main()