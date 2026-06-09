"""
Audio Processor - Core audio preprocessing
Handles loading, normalization, resampling
"""

import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Process audio files for Model 4 training"""
    
    def __init__(self, target_sr: int = 44100, mono: bool = False):
        self.target_sr = target_sr
        self.mono = mono
    
    def load_audio(self, audio_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """Load audio file with resampling"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=self.mono)
            logger.info(f"✓ Loaded: {Path(audio_path).name}")
            return audio, sr
        except Exception as e:
            logger.error(f"✗ Error loading {audio_path}: {e}")
            return None, None
    
    def normalize_loudness(self, audio: np.ndarray, target_loudness: float = -23.0) -> np.ndarray:
        """Normalize audio loudness (RMS-based)"""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-10:
            return audio
        
        current_db = 20 * np.log10(rms)
        gain_db = target_loudness - current_db
        gain_linear = 10 ** (gain_db / 20)
        
        return audio * gain_linear
    
    def save_audio(self, audio: np.ndarray, output_path: str, sr: int = 44100) -> bool:
        """Save audio file"""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, audio, sr)
            logger.info(f"✓ Saved: {output_path}")
            return True
        except Exception as e:
            logger.error(f"✗ Error saving {output_path}: {e}")
            return False
