# Оркестратор (Задача А)

Этот модуль отвечает за интеграцию всех компонентов системы Джарвис в единое приложение.

## Структура файлов

- `contracts.py` — интерфейсы между модулями (передайте участникам Б, Ц, Г)
- `app.py` — главный оркестратор с циклом обработки запросов
- `mocks_for_testing.py` — заглушки для локального тестирования
- `requirements.txt` — зависимости (пока только стандартная библиотека)

## Интеграция аудио-модуля (Задача Б)

Аудио-модуль находится в репозитории друга: https://github.com/sheelestun/Edge_NSU_b_part

### Шаг 1: Подключите как git submodule

```bash
cd Jarvis_from_Spartans
git submodule add https://github.com/sheelestun/Edge_NSU_b_part orchestrator/audio_module
git commit -m "Add audio module as submodule"
git push
```

### Шаг 2: Обновите app.py

В файле `orchestrator/app.py`:

1. Раскомментируйте импорт:
   ```python
   # from audio_module import AudioEngine  # <-- РАСКОММЕНТИРОВАТЬ
   ```

2. Замените заглушку в `__init__`:
   ```python
   # self.voice = AudioEngineMock()  # <-- ЗАКОММЕНТИРОВАТЬ
   self.voice = AudioEngine()        # <-- РАСКОММЕНТИРОВАТЬ
   ```

## Запуск

### Тестирование на заглушках (локально):

```bash
cd orchestrator
python app.py
```

### После интеграции реальных модулей:

```bash
cd orchestrator
python app.py
```

## Контракты

Все модули должны реализовать интерфейсы из `contracts.py`:

- **AudioModuleInterface** — для модуля Б (аудио, NPU, STT, TTS)
- **DatabaseModuleInterface** — для модуля Ц (БД, профили, RAG)
- **LLMModuleInterface** — для модуля Г (генерация ответов)
