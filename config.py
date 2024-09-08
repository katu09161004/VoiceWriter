from enum import Enum

SAMPLE_RATE = 16000
FRAME_LENGTH = 512
CONFIG_FILE = "voicewriter_config.json"

class ProcessingMode(Enum):
    API = "api"
    CPU = "cpu"
    CUDA = "cuda"

class WhisperModel(Enum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"