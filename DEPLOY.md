# Развёртывание «Джарвиса» на Firefly ROC-RK3588S-PC

Плата: 4 ГБ RAM (реально 3.8), 20 ГБ диска, **общая с другими командами**.
Поэтому на неё едет только то, что нужно в рантайме. Всё, что требует
торча ради экспорта весов, — остаётся на ноутбуке.

**Исключение — идентификация по голосу.** Пока `speaker.py` работает через
`wespeaker` + torch, они стоят на плате (~5 ГБ). Это временное решение,
см. раздел 8.

Ориентир по объёму: базовое окружение ~400 МБ, модели ~235 МБ,
wespeaker + torch ещё ~5 ГБ. **Перед установкой нужно ~7 ГБ свободного
места** (с запасом на распаковку колёс).

---

## 0. Что чем доставляется

| Что | Как попадает на плату | Размер |
|---|---|---|
| Код оркестратора, `llm_module.py`, подмодули Б и Ц | `git pull` | ~200 КБ |
| Голос Piper `ru-daniel_gleb-medium.onnx` + `.json` | **scp** (в `.gitignore`: `*.onnx`) | 61 МБ |
| Эмбеддинги RAG `models/` (модуль Ц) | **scp** (в `.gitignore`: `models/`) | 30 МБ |
| Whisper `ggml-base.bin` | **скачивается на плате** (curl) | 141 МБ |
| Модель openWakeWord `hey_jarvis` | **скачивается на плате** (python) | 3 МБ |
| Эталонные записи голосов `audio_dataset/` | **scp** (в `.gitignore`) | единицы МБ |
| `wespeaker` + torch | **pip на плате** (из git, не с PyPI) | ~5 ГБ |
| Предобученная модель WeSpeaker (`english`) | **скачивается при первом `import speaker`** | ~100 МБ |
| `paths.py`, `.env` | **создаются на плате** (пути локальные) | — |

Правило простое: **исходники — git, веса и приватное — scp, публичные
веса — curl прямо на плате** (незачем гонять 141 МБ через свой ноутбук).

---

## 1. Перед отъездом: что собрать на ноутбуке

Этим шагам нужны torch и transformers (~2.5 ГБ). `transformers` на плату
не едет никогда; torch там появится только ради wespeaker (раздел 4).

```bash
# эмбеддинги для RAG: 111 МБ fp32 -> 28 МБ int8, на плату едет int8
cd orchestrator/memory_module
python3 tools/export_embedding_model.py     # пишет models/
```

Голос Piper уже собран (`tts_voice/ru-daniel_gleb-medium.onnx`, 61 МБ).

Дальше — запушить код. Если менялись подмодули, то **сначала их, потом
main-репозиторий**:

```bash
git push --recurse-submodules=on-demand
```

Флаг делает это сам: пушит коммиты подмодулей, на которые ссылается
main, и только потом main. Обычный `git push` пушит лишь указатель, и
тогда на плате `git submodule update` падает с «not our ref» — коммита,
на который указывает main, на GitHub ещё нет.
---

## 2. Плата: доступ

```bash
sudo netbird up --management-url https://netbird.ci.nsu.ru --setup-key <ключ>
ssh firefly@100.107.17.63        # пароль: firefly
```

Домашний Wi-Fi до сервера может не пробиваться — нужен eduroam или мобильный.

---

## 3. Плата: conda-окружение

Conda на плате уже стоит (`~/miniconda3`). Своё окружение нужно, чтобы не
ломать базовое, в котором Фёдор запускает LLM-сервер.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
```

Эта строчка включает команду `conda activate` в текущей сессии. Если не
хочется писать её каждый раз — один раз выполнить `conda init bash` и
перезайти по ssh.

```bash
conda create -y -n jarvis python=3.11
```

**Почему 3.11, а не свежее.** Для aarch64 колёса `tflite-runtime`
(жёсткая зависимость openwakeword на Linux) собраны только до cp311.
На 3.12+ установка падает. Сам код работает через onnxruntime, а не
tflite, поэтому ниже мы её всё равно обходим — но 3.11 даёт запас.
Для wespeaker 3.11 тоже подходит: torch, torchaudio, numba и llvmlite
выпускают под неё aarch64-колёса.

```bash
conda activate jarvis
```

Дальше приглашение будет начинаться с `(jarvis)`. Всё остальное — в нём.

```bash
conda install -y -c conda-forge portaudio hdbscan
```

- `portaudio` — системная библиотека для `sounddevice` (микрофон и динамик).
  Через conda, а не `apt`: apt на плате регулярно занят
  `unattended-upgrades`, и sudo для conda не нужен.
- `hdbscan` — зависимость wespeaker. На PyPI для него **нет готовых колёс**,
  только исходники, и pip стал бы компилировать его на плате. conda-forge
  отдаёт уже собранный.

Если conda скажет, что `hdbscan` для этой платформы не нашёлся, — поставить
компилятор, тогда pip соберёт его сам (несколько минут):

```bash
conda install -y -c conda-forge compilers
```

---

## 4. Плата: Python-зависимости

```bash
pip install numpy onnxruntime sounddevice soundfile pywhispercpp piper-tts sqlite-vec tokenizers pyserial
```

Все пакеты ставятся готовыми aarch64-колёсами, ничего не компилируется.

```bash
pip install --no-deps openwakeword && pip install scipy scikit-learn tqdm requests
```

`--no-deps` — чтобы pip не потащил `tflite-runtime`. Проверено: с
`inference_framework="onnx"` openwakeword работает без него.

### wespeaker (идентификация по голосу, ~5 ГБ)

Сначала убедиться, что место есть — нужно ~7 ГБ свободных:

```bash
df -h ~
```

```bash
pip install --no-cache-dir torch torchaudio
```

Одной командой, чтобы pip подобрал **совместимую пару**: версия
`torchaudio` жёстко привязана к версии `torch`. С PyPI на aarch64 приезжает
CPU-сборка — CUDA-хвостов на ~2 ГБ, как на x86, не будет.

`--no-cache-dir` здесь важен: без него pip оставляет копию каждого
скачанного колеса в `~/.cache/pip`, и torch занимает место дважды.

```bash
pip install --no-cache-dir git+https://github.com/wenet-e2e/wespeaker.git
```

`wespeaker` **нет на PyPI** — `pip install wespeaker` и
`pip install -r audio_module/requirements.txt` падают с «No matching
distribution». Ставится только из git.

Он тянет `s3prl`, `openai-whisper`, `peft`, `accelerate`, `umap-learn`,
`numba`. Из них `s3prl` и `openai-whisper` на PyPI только исходниками, но
это чистый Python: «сборка» — просто упаковка, компилятор не нужен.
Единственный настоящий компилируемый пакет, `hdbscan`, уже поставлен
через conda в разделе 3.

По RAM: torch + модель WeSpeaker — по оценке порядка 0.5 ГБ резидентно
(не замерено; проверить через `free -h` при работающем `app.py`). Вместе
с LLM-сервером Фёдора и GNOME это впритык к 3.8 ГБ. Если плата начнёт
уходить в swap — это первое, что стоит убрать (раздел 8).

---

## 5. Плата: код и веса

```bash
cd ~/EDGE/24943/spartains && git clone --recurse-submodules <url> jarvis && cd jarvis
```

Уже клонированный репозиторий обновляется так:

```bash
git pull && git submodule update --init --recursive
```

Веса — с ноутбука (команды выполняются **на ноутбуке**):

```bash
scp tts_voice/ru-daniel_gleb-medium.* firefly@100.107.17.63:~/EDGE/24943/spartains/jarvis/tts_voice/
```

```bash
scp -r orchestrator/memory_module/models firefly@100.107.17.63:~/EDGE/24943/spartains/jarvis/orchestrator/memory_module/
```

```bash
scp -r orchestrator/audio_module/audio_dataset firefly@100.107.17.63:~/EDGE/24943/spartains/jarvis/orchestrator/audio_module/
```

Эталонные записи — это `audio_dataset/references/<имя>/*.wav`, по папке на
человека. Они в `.gitignore` модуля Б, поэтому живут только у того, кто их
записал. **Если папки пустые, ничего не падает — просто все говорящие
определяются как гость**, и персонализации не будет. Имя папки становится
`user_id`; если оно не совпадает с ключом в модуле памяти (`anton`, `masha`),
связать через `--user-map` (раздел 7).

Обратно **на плате** — публичные веса качаем напрямую:

```bash
mkdir -p orchestrator/audio_module/whisper.cpp && curl -L -o orchestrator/audio_module/whisper.cpp/ggml-base.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin
```

```bash
python -c "import openwakeword.utils as u; u.download_models(['hey_jarvis'])"
```

> `whisper.cpp` **компилировать не нужно**. `pywhispercpp` привозит свою
> сборку в колесе; из репозитория whisper.cpp нужен только файл модели.
> Инструкция про `cmake -B build` в README модуля Б к плате не относится.

Голос Piper модуль Б ищет под своим историческим именем:

```bash
cd orchestrator/audio_module && ln -sf ../../tts_voice/ru-daniel_gleb-medium.onnx ru_RU-dmitri-medium.onnx && ln -sf ../../tts_voice/ru-daniel_gleb-medium.json ru_RU-dmitri-medium.onnx.json && cd ../..
```

---

## 6. Плата: локальная настройка (файлы, которых нет в git)

**`orchestrator/audio_module/paths.py`** — где лежат эталонные записи:

```bash
cat > orchestrator/audio_module/paths.py <<'PY'
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

REFERENCE_DIR = BASE_DIR / "audio_dataset" / "references"
PY
```

**`orchestrator/memory_module/.env`** — путь к базе и ключи API.
Путь к базе обязательно абсолютный: иначе `jarvis.db` ищется относительно
текущего каталога, находится пустая база, и ассистент молча отвечает без
личных данных.

```bash
cat > orchestrator/memory_module/.env <<'ENV'
JARVIS_DB=/home/firefly/EDGE/24943/spartains/jarvis/orchestrator/memory_module/jarvis.db
JARVIS_HTTP_TIMEOUT=3
JARVIS_LAT=54.8430
JARVIS_LON=83.0920
WEATHERAPI_KEY=
DGIS_KEY=
ENV
```

Ключи можно оставить пустыми — погода и места тогда деградируют до
вежливой фразы, остальное работает. Координаты выше — Академгородок.

**Номер микрофона** в `orchestrator/audio_module/config.py` (`DEVICE_NUMBER`).
Список устройств:

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

**Наполнить базу:**

```bash
cd orchestrator/memory_module && python -m jarvis_memory.seed && cd ../..
```

Должно написать `vectors: built` — значит sqlite-vec подхватился и
семантический поиск включён.

---

## 7. Плата: проверка по нарастающей

Проверять по одному слою, а не всё сразу.

```bash
cd orchestrator && python check_integration.py
```

22 проверки логики, без железа и без LLM. Должно быть «Все проверки пройдены».

```bash
python app.py --mock --once
```

Весь цикл на заглушках.

```bash
curl http://localhost:8080/v1/models
```

Живой ли LLM-сервер Фёдора. Если нет — команда запуска в
`INSTRUCTIONS_FOR_DANIEL_DOING_THE_LLM_INTEGRATION.md`, раздел 4.

```bash
python llm_module.py "Привет! Кратко: кто ты?"
```

Оркестратор ↔ LLM напрямую, без микрофона.

```bash
cd audio_module && python -c "import speaker; print({k: len(v) for k, v in speaker.REFERENCE_EMBEDDINGS.items()})" && cd ..
```

Идентификация по голосу отдельно. Первый запуск **скачивает модель
WeSpeaker из интернета** (нужен выход в сеть с платы) и считает эмбеддинги
всех эталонных записей — это может занять минуту-другую. Должно напечатать
`{'имя': число_записей, ...}`. Пустой `{}` значит, что в
`audio_dataset/references/` нет записей: всё будет работать, но все — гости.

Делать это до полного прогона стоит, чтобы скачивание не случилось
посреди демонстрации: `speaker.py` грузит модель при импорте.

```bash
python app.py --once --log-level DEBUG
```

Полный прогон на железе. Связать ID говорящего с профилями памяти:
`--user-map shelestov=anton,puchkov=masha`. Без платы индикатора —
добавить `--no-indicator`.

---

## 8. Технический долг: wespeaker на плате

Сейчас выбран самый быстрый путь: `wespeaker` + torch ставятся на плату
как есть (раздел 4). Работает, но стоит ~5 ГБ диска общей платы и
заметную долю RAM, а нужен от всего этого ровно один вызов — «аудио →
вектор».

Как это убрать потом, не меняя контракт `identify_speaker(audio) -> dict`:

- Эмбеддер WeSpeaker экспортируется в ONNX **на ноутбуке**.
- Эмбеддинги эталонных записей считаются там же и едут на плату одним
  файлом `.npz` — сейчас они пересчитываются при каждом старте.
- На плате остаётся только onnxruntime, который уже стоит.
- Освобождается: `pip uninstall torch torchaudio wespeaker s3prl openai-whisper peft accelerate`.

Решение и реализация — за Даниилшем, это его модуль.
