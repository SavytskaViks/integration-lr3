from flask import Flask, render_template, request, jsonify
import torch
import torchaudio
import torch.nn.functional as F
from model import SpeechCommandModel
import time
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["up", "down", "yes", "no"]

# Шлях до моделі: з env або за замовчуванням best_model.pth
MODEL_PATH = os.getenv("MODEL_PATH", "best_model.pth")

# Завантажуємо модель
model = SpeechCommandModel(num_classes=len(CLASSES)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Розмір моделі
model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html", model_size=f"{model_size_mb:.2f} MB")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    # Завантажуємо аудіо
    waveform, sample_rate = torchaudio.load(file.stream, normalize=True)

    # Перетворюємо у спектрограму
    transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=64)
    spec = transform(waveform)  # [1, n_mels, time]

    # Padding до фіксованої ширини
    desired_len = 64
    if spec.shape[2] < desired_len:
        spec = F.pad(spec, (0, desired_len - spec.shape[2]))
    elif spec.shape[2] > desired_len:
        spec = spec[:, :, :desired_len]

    spec = spec.unsqueeze(0).to(DEVICE)  # [1, 1, n_mels, time]

    # Вимірюємо latency
    start_time = time.time()
    with torch.no_grad():
        outputs = model(spec)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    end_time = time.time()
    latency_ms = int((end_time - start_time) * 1000)

    pred_index = probs.argmax()
    return jsonify({
        "prediction": CLASSES[pred_index],
        "probabilities": {cls: float(f"{p * 100:.1f}") for cls, p in zip(CLASSES, probs)},
        "latency_ms": latency_ms
    })

if __name__ == "__main__":
    # Для локального запуску та докера
    app.run(host="0.0.0.0", port=8000, debug=True)
