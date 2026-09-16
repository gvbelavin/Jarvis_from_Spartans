from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable


GUEST_NAME = "Гость"


@dataclass
class SpeakerResult:
    user_id: Optional[str]
    user_name: str
    confidence: float


@runtime_checkable
class AudioModuleInterface(Protocol):
    async def wait_for_wake_word(self) -> None:
        ...

    async def record_phrase(self) -> Any:
        ...

    async def identify_speaker(
        self,
        audio_data: Any,
    ) -> SpeakerResult:
        ...

    async def speech_to_text(
        self,
        audio_data: Any,
    ) -> str:
        ...

    async def play_tts(
        self,
        text: str,
    ) -> None:
        ...

    async def listen_once(self) -> tuple[SpeakerResult, str]:
        ...


@runtime_checkable
class DatabaseModuleInterface(Protocol):
    def build_context(
        self,
        user_id: Optional[str],
        transcript: str,
    ) -> Any:
        ...

    def record_answer(
        self,
        user_id: Optional[str],
        transcript: str,
        answer: str,
    ) -> None:
        ...


@runtime_checkable
class LLMModuleInterface(Protocol):
    async def generate(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        ...