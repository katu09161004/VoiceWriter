from typing import List, Tuple
import tempfile
import wave
import os
import struct
import asyncio
import concurrent.futures
import torch
from torch import Tensor
import whisper
from openai import OpenAI
from config import SAMPLE_RATE, WhisperModel

class TranscriptionProcessor:
    async def process(self, frames: List[Tuple[int, ...]]) -> str:
        raise NotImplementedError("This method should be implemented by subclasses")

class APITranscriptionProcessor(TranscriptionProcessor):
    def __init__(self, apiKey: str) -> None:
        self.client = OpenAI(api_key=apiKey)

    async def process(self, frames: List[Tuple[int, ...]]) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tempAudioFile:
            waveFile = wave.open(tempAudioFile, 'wb')
            waveFile.setnchannels(1)
            waveFile.setsampwidth(2)
            waveFile.setframerate(SAMPLE_RATE)
            waveFile.writeframes(b''.join(struct.pack("h" * len(frame), *frame) for frame in frames))
            waveFile.close()
        
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(pool, self._transcribe, tempAudioFile.name)
        
        os.unlink(tempAudioFile.name)
        return result.text

    def _transcribe(self, file_path):
        with open(file_path, "rb") as audioFile:
            return self.client.audio.transcriptions.create(model="whisper-1", file=audioFile)

class LocalTranscriptionProcessor(TranscriptionProcessor):
    def __init__(self, useCuda: bool, model: WhisperModel) -> None:
        self.useCuda = useCuda
        self.modelType = model
        self.model = None

    def loadModel(self):
        device = "cuda" if self.useCuda and torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(self.modelType.value, device=device)

    async def process(self, frames: List[Tuple[int, ...]]) -> str:
        audioArray: Tensor = torch.tensor(frames, dtype=torch.float32).flatten() / 32768.0
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(pool, self.model.transcribe, audioArray)
        return result["text"]