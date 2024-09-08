import tkinter as tk
from tkinter import ttk
import pyaudio
import whisper
import torch
from torch import Tensor
import struct
import wave
import locale
import os
import keyboard
import pyautogui
import pyperclip
import tempfile
from typing import List, Tuple, Optional
from enum import Enum
from openai import OpenAI
import threading
import json
import asyncio
import concurrent.futures

# 処理モードの定義
class ProcessingMode(Enum):
    API = "api"
    CPU = "cpu"
    CUDA = "cuda"

# Whisperモデルの定義
class WhisperModel(Enum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

# 設定
SAMPLE_RATE = 16000
FRAME_LENGTH = 512
CONFIG_FILE = "voicewriter_config.json"

class AudioProcessor:
    def __init__(self) -> None:
        self.pa = pyaudio.PyAudio()
        self.audioStream: Optional[pyaudio.Stream] = None
        self.shortcutKey: str = ""

    def initialize_stream(self, inputDeviceIndex: int) -> None:
        self.audioStream = self.pa.open(
            rate=SAMPLE_RATE,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            input_device_index=inputDeviceIndex,
            frames_per_buffer=FRAME_LENGTH
        )

    def get_input_devices(self) -> List[str]:
        deviceCount = self.pa.get_device_count()
        devices = []
        for i in range(deviceCount):
            deviceInfo = self.pa.get_device_info_by_index(i)
            if deviceInfo["maxInputChannels"] > 0:
                try:
                    deviceName = deviceInfo['name'].encode(locale.getpreferredencoding()).decode('utf-8')
                except (UnicodeEncodeError, UnicodeDecodeError):
                    deviceName = deviceInfo['name']
                if deviceName not in devices:  # 重複を削除
                    devices.append(deviceName)
        return devices

    def record_audio(self) -> List[Tuple[int, ...]]:
        frames: List[Tuple[int, ...]] = []
        while keyboard.is_pressed(self.shortcutKey):  # ショートカットキーが押されている間だけ録音
            pcm = self.audioStream.read(FRAME_LENGTH)
            pcm = struct.unpack_from("h" * FRAME_LENGTH, pcm)
            frames.append(pcm)
        return frames

    def close(self) -> None:
        if self.audioStream is not None:
            self.audioStream.close()
        self.pa.terminate()

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
            waveFile.setsampwidth(2)  # 16-bit audio
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

class VoiceWriterGUI:
    def __init__(self, master):
        self.master = master
        master.title("VoiceWriter")
        
        self.audioProcessor = AudioProcessor()
        self.transcriptionProcessor = None
        self.isRecording = False

        # 設定の読み込み
        self.loadConfig()

        # 処理モード選択
        ttk.Label(master, text="処理モード:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.processingMode = tk.StringVar(value=self.config.get("processingMode", "cpu"))
        self.processingModeCombo = ttk.Combobox(master, textvariable=self.processingMode, values=["api", "cpu", "cuda"], state="readonly")
        self.processingModeCombo.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.processingModeCombo.bind("<<ComboboxSelected>>", self.onProcessingModeChange)

        # APIキー入力
        ttk.Label(master, text="APIキー:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.apiKey = tk.StringVar(value=self.config.get("apiKey", ""))
        self.apiKeyEntry = ttk.Entry(master, textvariable=self.apiKey, show="*")
        self.apiKeyEntry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        self.apiKeyEntry.config(state="normal" if self.processingMode.get() == "api" else "disabled")

        # Whisperモデル選択
        ttk.Label(master, text="Whisperモデル:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.whisperModel = tk.StringVar(value=self.config.get("whisperModel", "medium"))
        self.whisperModelCombo = ttk.Combobox(master, textvariable=self.whisperModel, values=[model.value for model in WhisperModel], state="readonly")
        self.whisperModelCombo.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        self.whisperModelCombo.config(state="normal" if self.processingMode.get() != "api" else "disabled")

        # マイク選択
        ttk.Label(master, text="マイク:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.selectedMic = tk.StringVar(value=self.config.get("selectedMic", ""))
        self.micCombo = ttk.Combobox(master, textvariable=self.selectedMic, values=self.audioProcessor.get_input_devices(), state="readonly")
        self.micCombo.grid(row=3, column=1, sticky="ew", padx=5, pady=5)

        # ショートカットキー入力
        ttk.Label(master, text="ショートカットキー:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.shortcutKey = tk.StringVar(value=self.config.get("shortcutKey", "ctrl+space"))
        self.shortcutKeyEntry = ttk.Entry(master, textvariable=self.shortcutKey)
        self.shortcutKeyEntry.grid(row=4, column=1, sticky="ew", padx=5, pady=5)

        # 認識開始/終了ボタン
        self.recognitionButton = ttk.Button(master, text="認識開始", command=self.toggleRecognition)
        self.recognitionButton.grid(row=5, column=0, columnspan=2, pady=10)

        # ログ表示エリア
        ttk.Label(master, text="ログ:").grid(row=6, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        self.logText = tk.Text(master, height=10, width=50, state="disabled")
        self.logText.grid(row=7, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

        # 最新の文字起こし結果表示エリア
        ttk.Label(master, text="最新の文字起こし結果:").grid(row=8, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        self.latestTranscriptionText = tk.Text(master, height=10, width=50)
        self.latestTranscriptionText.grid(row=9, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

        # ウィンドウサイズの調整
        master.columnconfigure(1, weight=1)
        master.rowconfigure(7, weight=1)
        master.rowconfigure(9, weight=1)

        # ウィンドウが閉じられる時の処理を設定
        master.protocol("WM_DELETE_WINDOW", self.onClose)

        # イベントループの設定
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.run_async_loop, daemon=True)
        self.thread.start()

    def run_async_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def loadConfig(self):
        try:
            with open(CONFIG_FILE, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = {}

    def saveConfig(self):
        config = {
            "processingMode": self.processingMode.get(),
            "apiKey": self.apiKey.get(),
            "whisperModel": self.whisperModel.get(),
            "selectedMic": self.selectedMic.get(),
            "shortcutKey": self.shortcutKey.get()
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)

    def onClose(self):
        self.saveConfig()
        self.master.destroy()

    def onProcessingModeChange(self, event):
        if self.processingMode.get() == "api":
            self.apiKeyEntry.config(state="normal")
            self.whisperModelCombo.config(state="disabled")
        else:
            self.apiKeyEntry.config(state="disabled")
            self.whisperModelCombo.config(state="readonly")

    def toggleRecognition(self):
        if not self.isRecording:
            self.startRecognition()
        else:
            self.stopRecognition()

    def startRecognition(self):
        self.isRecording = True
        self.recognitionButton.config(text="認識終了")
        self.logMessage("認識を開始します...")

        # 設定項目を無効化
        self.processingModeCombo.config(state="disabled")
        self.apiKeyEntry.config(state="disabled")
        self.whisperModelCombo.config(state="disabled")
        self.micCombo.config(state="disabled")
        self.shortcutKeyEntry.config(state="disabled")

        # 選択されたマイクのインデックスを取得
        selectedMicName = self.selectedMic.get()
        inputDeviceIndex = self.audioProcessor.get_input_devices().index(selectedMicName)
        self.audioProcessor.initialize_stream(inputDeviceIndex)

        # ショートカットキーを設定
        self.audioProcessor.shortcutKey = self.shortcutKey.get()

        # TranscriptionProcessorの初期化
        if self.processingMode.get() == "api":
            apiKey = self.apiKey.get()
            maskedApiKey = apiKey[:4] + '*' * (len(apiKey) - 8) + apiKey[-4:]
            self.transcriptionProcessor = APITranscriptionProcessor(apiKey)
            self.logMessage(f"APIキー: {maskedApiKey}")
        else:
            useCuda = self.processingMode.get() == "cuda"
            selectedModel = WhisperModel(self.whisperModel.get())
            self.transcriptionProcessor = LocalTranscriptionProcessor(useCuda, selectedModel)
            self.logMessage(f"モデル '{selectedModel.value}' をロード中です。しばらくお待ちください...")
            threading.Thread(target=self.loadModelInBackground).start()

        # ショートカットキーの監視を開始
        self.checkShortcutAndRecord()

    def loadModelInBackground(self):
        self.transcriptionProcessor.loadModel()
        self.logMessage("モデルのロードが完了しました。")

    def stopRecognition(self):
        self.isRecording = False
        self.recognitionButton.config(text="認識開始")
        self.logMessage("認識を終了しました")

        # 設定項目を有効化
        self.processingModeCombo.config(state="readonly")
        self.apiKeyEntry.config(state="normal" if self.processingMode.get() == "api" else "disabled")
        self.whisperModelCombo.config(state="readonly" if self.processingMode.get() != "api" else "disabled")
        self.micCombo.config(state="readonly")
        self.shortcutKeyEntry.config(state="normal")

    def checkShortcutAndRecord(self):
        if self.isRecording:
            if keyboard.is_pressed(self.shortcutKey.get()):
                self.logMessage("録音中...")
                frames = self.audioProcessor.record_audio()
                self.logMessage("録音終了")
                
                if frames:  # フレームが存在する場合のみ処理を行う
                    self.logMessage("文字起こしを開始します...")
                    future = asyncio.run_coroutine_threadsafe(self.processTranscription(frames), self.loop)
                    future.add_done_callback(self.handle_transcription_result)
            
            self.master.after(100, self.checkShortcutAndRecord)

    def handle_transcription_result(self, future):
        try:
            result = future.result()
            self.master.after(0, self.update_gui, result)
        except Exception as e:
            self.master.after(0, self.logMessage, f"エラーが発生しました: {str(e)}")

    def update_gui(self, transcribedText):
        self.logMessage(f"文字起こし結果: {transcribedText}")
        
        # 最新の文字起こし結果を表示
        self.latestTranscriptionText.delete('1.0', tk.END)
        self.latestTranscriptionText.insert(tk.END, transcribedText)
        
        pyperclip.copy(transcribedText)
        pyautogui.hotkey('ctrl', 'v')
        activeWindow = pyautogui.getActiveWindow()
        self.logMessage(f"文字起こし結果を '{activeWindow.title}' に貼り付けました")

    async def processTranscription(self, frames):
        return await self.transcriptionProcessor.process(frames)

    def logMessage(self, message):
        self.logText.config(state="normal")
        self.logText.insert(tk.END, message + "\n")
        self.logText.see(tk.END)
        self.logText.config(state="disabled")
        self.master.update_idletasks()  # GUIを即時更新

def main():
    root = tk.Tk()
    gui = VoiceWriterGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()