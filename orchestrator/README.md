# Оркестратор (Задача А)

Модуль интеграции: связывает аудиомодуль (Б), модуль памяти (Ц) и LLM (Г)
в единое приложение «Джарвис».

## Структура

| Файл | Назначение |
|---|---|
| `app.py` | Главный цикл: wake word → STT → память → LLM → TTS |
| `contracts.py` | Контракты интерфейсов между модулями |
| `settings.py` | Настройки оркестратора (пути к подмодулям, таймаут записи) |
| `audio_adapter.py` | Тонкий адаптер над git submodule `audio_module` (модуль Б) |
| `llm_adapter.py` | Обёртка локальной LLM к контракту `generate(messages)` (модуль Г) |
| `llm_module.py` | HTTP-клиент RKLLM-сервера (`localhost:8080/v1/chat/completions`) |
| `mocks_for_testing.py` | Заглушки для запуска без железа |
| `check_integration.py` | Офлайн-проверка связки (память, адаптеры, клиент LLM) |
| `indicator.py` | LED-индикатор + кнопка mute на ESP32 (модуль 5) |
| `indicating_audio.py` | Обёртка над audio, шлющая состояния на индикатор |
| `requirements.txt` | Зависимости оркестратора |

Модули Б и Ц подключены **git submodule**:

```
orchestrator/audio_module/    → sheelestun/Edge_NSU_b_part   (модуль Б)
orchestrator/memory_module/   → LessVegetables/jarvis-memory (модуль Ц)
```

### Почему `settings.py`, а не `config.py`

Подмодуль аудио содержит собственный `config.py` и импортирует его как
top-level (`import config`), поэтому `audio_adapter.py` добавляет папку
`audio_module` в `sys.path`. Лежи настройки оркестратора в `config.py`,
возникла бы коллизия имён: `import config` резолвился бы в зависимости от
порядка импортов. Имя `settings` исключает это.

Параметры аудиомодуля (`SPEAKER_THRESHOLD`, `SAMPLE_RATE`, `CHUNK_SIZE`,
`SILENCE_CHUNKS`) в оркестраторе **не дублируются**: единственный источник
правды — `audio_module/config.py`, адаптер читает их оттуда.

### Известное расхождение в подмодуле Б

В рабочей копии `audio_module` есть **незакоммиченная** правка `config.py`:

```diff
-SILENCE_CHUNKS = 13                                          # было (эффективное значение)
+SILENCE_CHUNKS = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE) + 1   # стало 7
```

Один блок = 80 мс, поэтому окно тишины, по которому модуль Б завершает
запись, изменилось примерно с **1.04 с на 0.56 с**. Это правка алгоритма
чужого модуля: команды могут обрываться на паузе внутри фразы. Расхождение
оставлено как есть осознанно — вернуть прежнее поведение можно, вернув
`SILENCE_CHUNKS = 13`.

## Реальные интерфейсы (сверено с кодом, не с ТЗ)

### Модуль Б — `audio_module`

**Класса `AudioEngine` и функции `run_pipeline()` в нём нет.**
`pipeline.py::main()` — это готовый сценарий целиком: печатает результат,
подставляет ответ `"Я получил твою команду."` и возвращает `None`,
поэтому как функция он не переиспользуется.

Фактические точки входа:

| Модуль | Вызов | Возвращает |
|---|---|---|
| `audio.py` | `start_audio(callback)` | `sd.InputStream` (**не запущенный**) |
| `wake_word.py` | `detect(np.ndarray[int16])` | `bool` |
| `comand_record.py` | `CommandRecorder` | `.process()`, `.finished`, `.get_audio()` |
| `speaker.py` | `identify_speaker(audio)` | `dict{user_id, user_name, confidence}` |
| `stt.py` | `transcribe(audio)` | `str` |
| `tts.py` | `synthesize(text)` | `bytes` (WAV в памяти) |
| `player.py` | `play_audio(bytes)` | `None` (блокирующий) |

`AudioAdapter` вызывает эти функции по отдельности в том же порядке, что и
`pipeline.py`, — без его печати и его заглушки ответа. Ни один файл
подмодуля не редактировался.

Что адаптер добавляет сверху:

- переводит блокирующие вызовы в `asyncio.to_thread`, чтобы не вешать event loop;
- держит **один** `InputStream` от wake word до конца записи (как `pipeline.py`):
  переоткрытие потока теряет начало команды;
- страховочный таймаут записи (`--record-timeout`): сам модуль Б завершает
  запись только по тишине и может ждать её бесконечно;
- приводит `dict` от `identify_speaker()` к `SpeakerResult`, включая случай
  `user_name=None` (голос не опознан);
- `--user-map` связывает ID говорящего с профилем памяти.

### Модуль Ц — `jarvis_memory`

Реализует публичный API ровно как заявлено:

```python
import jarvis_memory as memory

ctx = memory.build_context(user_id, transcript)   # user_id=None — гость
answer = await llm.generate(ctx.to_messages())
memory.record_answer(user_id, transcript, answer)
```

`user_id=None` поддержан самим модулем: он отдаёт нейтральный контекст и
запрещает раскрывать личные данные. Ничего дописывать не потребовалось.

### Модуль Г — LLM (RKLLM-сервер на плате)

Контракт один: `await llm.generate(messages: list[dict[str, str]]) -> str`.
Системный промпт и вопрос отдельно **не** передаются — они уже внутри
`ctx.to_messages()`.

Модуль Г поставлен не Python-пакетом, а HTTP-сервером Фёдора:
пропатченный `flask_server.py` из `rknn-llm` слушает **порт 8080**
(`~/rkllm_server/` на плате). `llm_module.LLMEngine` шлёт туда тот же
JSON, что в инструкции:

```python
# эквивалент того, что делает LLMEngine.generate(ctx.to_messages())
{"model": "rkllm", "messages": ctx.to_messages(), "stream": False}
# POST http://localhost:8080/v1/chat/completions
```

`localhost` — потому что оркестратор и сервер живут на одной плате.
VPN нужен только чтобы зайти на плату со своего компьютера.

`LLMAdapter` сам определяет, синхронный движок или асинхронный.
**Синхронный вызывается через `asyncio.to_thread`** — иначе генерация
на ~3 с заблокировала бы event loop. Пока модель думает, оркестратор
ставит индикатор в `thinking` и пишет в лог «Думаю...».

Переопределение без правки кода: `JARVIS_LLM_URL`, `JARVIS_LLM_MODEL`,
`JARVIS_LLM_TIMEOUT`.

## Запуск

### Офлайн, без железа и LLM (проверка связки)

```bash
cd orchestrator
python check_integration.py     # офлайн-проверка связки, без железа
python app.py --mock --once     # один прогон на заглушках
```

### На реальном железе

Перед запуском настроить модуль Б по его README: собрать `whisper.cpp`,
положить `ggml-base.bin`, модель Piper и эталонные записи в
`audio_module/audio_dataset/references/<user_id>/`, создать `paths.py`
(он в `.gitignore` подмодуля), выставить `DEVICE_NUMBER` в `audio_module/config.py`.

`app.py` проверяет готовность модуля Б **до** первого обращения к железу и
вместо сырого `ModuleNotFoundError` печатает список того, чего не хватает.
Если чего-то нет, запуск завершается с кодом 2 и инструкцией.

```bash
cd orchestrator

# живой LLM-сервер на localhost:8080 подхватится сам (llm_module.LLMEngine)
curl http://localhost:8080/v1/models
python llm_module.py "Привет! Кратко: кто ты?"
python app.py

# другой движок, если понадобится
python app.py --llm-module my_llm --llm-class MyEngine

# связать ID говорящего модуля Б с профилем памяти
python app.py --user-map shelestov=anton,puchkov=masha

# прочие ключи
python app.py --device 2 --record-timeout 8 --once --log-level DEBUG
```

Если `llm_module.py` удалить, оркестратор **не падает**: отвечает
фразой-заглушкой. Если файл на месте, а сервер на :8080 выключен —
команда обработки ловит ошибку, говорит «произошла ошибка» и снова
слушает wake word. Команда подъёма сервера есть в логе `LLMServerError`.

## Важное про ID говорящего

Модуль Б берёт `user_id` из имени папки с эталонными записями
(`audio_dataset/references/shelestov/`), а модуль памяти знает ключи
`anton` / `masha`. Если это разные строки, личные данные молча не
подтянутся. `app.py` предупреждает об этом в логе, а связывает ID
ключ `--user-map`.

### Модуль 5 — индикатор на ESP32

ESP32 подключён по USB и показывает на LED текущее состояние оркестратора
(`idle`, `listening`, `thinking`, `speaking`, `guest`, `error`), плюс кнопка
BOOT на плате работает как мьют микрофона: пока mute=on, оркестратор не
запускает wake word и не открывает микрофон.

Обёртка `IndicatingAudioAdapter` вставляет `set_state()` в те точки внутри
`listen_once()`, которые снаружи не видны (между wake word и записью,
между записью и STT). `AudioAdapter` и `AudioEngineMock` при этом не
редактируются.

Если платы нет — `IndicatorEsp32.open()` молча деградирует до заглушки,
пайплайн работает как раньше. Явное отключение — `--no-indicator`.

```bash
python app.py --mock --once                          # с реальной платой (если есть)
python app.py --mock --once --no-indicator           # заведомо без платы
python app.py --mock --once --indicator-port /dev/ttyUSB1
```

Прошивка и Python-класс живут в отдельном репозитории Степана; сюда
попадает только клиентская обёртка (`indicator.py`) и обёртка audio.

## Ограничения

- Для гостя история диалога в `jarvis_memory` общая на всех
  неизвестных говорящих (известная особенность модуля Ц).
- Модуль Б не имеет VAD в полном смысле: конец фразы определяется по
  средней амплитуде. Это особенность подмодуля, не изменялась.