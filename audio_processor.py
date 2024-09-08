import pyaudio
import keyboard
import struct
import locale
from typing import List, Tuple, Optional

SAMPLE_RATE = 16000
FRAME_LENGTH = 512

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
        while keyboard.is_pressed(self.shortcutKey):
            pcm = self.audioStream.read(FRAME_LENGTH)
            pcm = struct.unpack_from("h" * FRAME_LENGTH, pcm)
            frames.append(pcm)
        return frames

    def close(self) -> None:
        if self.audioStream is not None:
            self.audioStream.close()
        self.pa.terminate()