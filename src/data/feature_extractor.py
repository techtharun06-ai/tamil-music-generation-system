"""
Feature Extractor - Extract music features from audio
BPM, key detection, chroma, MFCC, F0 contour, etc.
"""

import librosa
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extract music features from audio"""
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
    
    def extract_bpm(self, audio: np.ndarray) -> float:
        """Extract BPM using onset strength"""
        try:
            onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
            bpm = librosa.beat.tempo(onset_strength=onset_env, sr=self.sr)[0]
            return float(bpm)
        except Exception as e:
            logger.warning(f"BPM extraction failed: {e}")
            return 120.0
    
    def extract_chroma(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract chroma features (key detection)"""
        try:
            chroma_cqt = librosa.feature.chroma_cqt(y=audio, sr=self.sr)
            chroma_stft = librosa.feature.chroma_stft(y=audio, sr=self.sr)
            
            return {
                'chroma_cqt': chroma_cqt,
                'chroma_stft': chroma_stft,
                'chroma_mean': chroma_cqt.mean(axis=1),
                'chroma_std': chroma_cqt.std(axis=1)
            }
        except Exception as e:
            logger.warning(f"Chroma extraction failed: {e}")
            return {}
    
    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """Extract MFCC features"""
        try:
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=n_mfcc)
            return mfcc
        except Exception as e:
            logger.warning(f"MFCC extraction failed: {e}")
            return np.array([])
    
    def extract_all(self, audio: np.ndarray, duration: Optional[float] = None) -> Dict:
        """Extract all features from audio"""
        if duration is None:
            duration = len(audio) / self.sr
        
        features = {
            'duration': duration,
            'sr': self.sr,
            'n_samples': len(audio),
            'rms': float(np.sqrt(np.mean(audio ** 2))),
            'bpm': self.extract_bpm(audio),
            'chroma': self.extract_chroma(audio),
            'mfcc': self.extract_mfcc(audio)
        }
        
        return features
