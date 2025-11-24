const predictBtn = document.getElementById('predictBtn');
const resetBtn = document.getElementById('resetBtn');
const fileInput = document.getElementById('audioFile');
const fileLabel = document.getElementById('fileLabel');
const predElem = document.getElementById('prediction');
const latencyElem = document.getElementById('latency');
const probsElem = document.getElementById('probabilities');
const fileLabelText = fileLabel.querySelector('span');

// 🟢 Оновлення назви файлу після вибору
fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) {
    fileLabelText.textContent = fileInput.files[0].name;
  } else {
    fileLabelText.textContent = "Оберіть аудіофайл";
  }
});

// 🟢 Скидання файлу вручну (кнопка "Скинути файл")
resetBtn.addEventListener('click', () => {
  fileInput.value = '';
  fileLabelText.textContent = "Оберіть аудіофайл";
  predElem.textContent = "Прогноз: -";
  latencyElem.textContent = "Latency: - мс";
  probsElem.innerHTML = '';
});

// 🟢 Обробка натискання кнопки "Розпізнати"
predictBtn.addEventListener('click', async () => {
  if (fileInput.files.length === 0) {
    alert("Оберіть аудіофайл!");
    return;
  }

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  // Встановлюємо стан "завантаження"
  predElem.textContent = "Прогноз: ...";
  latencyElem.textContent = "Latency: ... мс";
  probsElem.innerHTML = '';

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error("Помилка сервера при розпізнаванні");
    }

    const data = await response.json();

    // 🟢 Виводимо прогноз
    predElem.textContent = `Прогноз: ${data.prediction}`;

    // 🟢 Вірогідності
    probsElem.innerHTML = '';
    for (let cls in data.probabilities) {
      const div = document.createElement('div');
      div.className = 'probability';
      div.textContent = `${cls}: ${data.probabilities[cls]}%`;

      if (cls === data.prediction) div.classList.add('green');
      else div.classList.add('red');
      probsElem.appendChild(div);
    }

    // 🟢 Latency
    latencyElem.textContent = `Latency: ${data.latency_ms} мс`;

  } catch (err) {
    alert("Сталася помилка: " + err.message);
    predElem.textContent = "Прогноз: -";
    latencyElem.textContent = "Latency: - мс";
    probsElem.innerHTML = '';
  } finally {
    // ✅ Дозволяємо повторно вибрати той самий файл
    fileInput.value = '';
    fileLabelText.textContent = "Оберіть аудіофайл";
  }
});

