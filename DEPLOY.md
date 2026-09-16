# Развёртывание «Джарвиса» на Firefly ROC-RK3588S-PC

Плата: 4 ГБ RAM (реально 3.8), 20 ГБ диска, **общая с другими командами**.
Поэтому на неё едет только то, что нужно в рантайме. Всё, что требует
компиляции, торча и гигабайтов — остаётся на ноутбуке.

Ориентир по объёму: окружение ~400 МБ, модели ~235 МБ. Итого меньше 700 МБ.

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
| `paths.py`, `.env` | **создаются на плате** (пути локальные) | — |

Правило простое: **исходники — git, веса и приватное — scp, публичные
веса — curl прямо на плате** (незачем гонять 141 МБ через свой ноутбук).

---

## 1. Перед отъездом: что собрать на ноутбуке

Это единственные шаги, которым нужны torch/transformers (~2.5 ГБ).
На плате они не нужны никогда.

```bash
# эмбеддинги для RAG: 111 МБ fp32 -> 28 МБ int8, на плату едет int8
cd orchestrator/memory_module
python3 tools/export_embedding_model.py     # пишет models/
```

Голос Piper уже собран (`tts_voice/ru-daniel_gleb-medium.onnx`, 61 МБ).

Дальше — запушить код:

```bash
git push
```

> **Важно про подмодуль Б.** В `orchestrator/audio_module` лежат два
> исправления (см. раздел 7). Репозиторий чужой — `sheelestun/Edge_NSU_b_part`.
> Пока они не влиты, указатель подмодуля в main-репозитории на них ставить
> НЕЛЬЗЯ: на плате `git submodule update` не найдёт этих коммитов.
> До вливания правки накатываются на плате патчами из `patches/`.

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

```bash
conda activate jarvis
```

Дальше приглашение будет начинаться с `(jarvis)`. Всё остальное — в нём.

```bash
conda install -y -c conda-forge portaudio
```

Это системная библиотека для `sounddevice` (микрофон и динамик). Ставим
через conda, а не `apt`, потому что apt на плате регулярно занят
`unattended-upgrades`, и sudo для conda не нужен.

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

**Чего здесь намеренно нет: `torch`, `torchaudio`, `transformers`,
`wespeaker`.** Это инструменты обучения и экспорта, им место на ноутбуке.
`wespeaker` вдобавок тянет `s3prl`, `openai-whisper`, `peft`, `accelerate`
— около 5 ГБ на диск, где всего 20 и живут ещё три команды.
Про идентификацию по голосу — раздел 8.

---

## 5. Плата: код и веса

```bash
cd ~/EDGE/24943/spartains && git clone --recurse-submodules <url> jarvis && cd jarvis
```

Уже клонированный репозиторий обновляется так:

```bash
git pull && git submodule update --init --recursive
```

Правки модуля Б, пока они не влиты в репозиторий Даниилша:

```bash
cd orchestrator/audio_module && git apply ../../patches/*.patch && cd ../..
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
python app.py --once --log-level DEBUG
```

Полный прогон на железе. Связать ID говорящего с профилями памяти:
`--user-map shelestov=anton,puchkov=masha`. Без платы индикатора —
добавить `--no-indicator`.

---

## 8. Открытый вопрос: идентификация по голосу

`speaker.py` модуля Б работает через `wespeaker` + torch, а `wespeaker`
нет на PyPI (только git) и он тянет ~5 ГБ. На плату в таком виде он не
поедет.

Сначала посмотреть, сколько вообще свободно:

```bash
df -h ~
```

Варианты, от дешёвого к дорогому:

1. **Запустить цепочку без идентификации.** Все говорящие — «гость»;
   модуль Ц это штатно поддерживает (`user_id=None`) и личных данных не
   раскрывает. Проверяется весь тракт микрофон → ответ, кроме персонализации.
   Нужен небольшой флаг в оркестраторе — сейчас `check_ready()` не даст
   стартовать без `wespeaker`.
2. **ONNX вместо torch.** Эмбеддер экспортируется в ONNX на ноутбуке,
   эталонные эмбеддинги считаются там же и едут на плату файлом `.npz`.
   На плате — только onnxruntime, который уже стоит. Контракт
   `identify_speaker(audio) -> dict` не меняется. Это же первый шаг к
   RKNN по ТЗ.
3. **Поставить wespeaker на плату как есть.** Честно работает, но ~5 ГБ,
   долгая установка и риск, что `hdbscan`/`umap-learn` не соберутся.

Решение за Даниилшем — это его модуль.
