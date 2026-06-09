"""
LRC Parser - Parse synchronized lyrics files
"""

import re
from typing import List, Dict, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class LRCParser:
    """Parse LRC (Lyric Time Codes) files"""
    
    @staticmethod
    def parse(lrc_path: str) -> Optional[List[Dict]]:
        """
        Parse LRC file and return list of (timestamp, lyric) entries
        
        LRC Format:
        [mm:ss.xx]lyric text
        [00:12.00]First line
        [00:17.20]Second line
        """
        lyrics = []
        
        try:
            if not Path(lrc_path).exists():
                logger.warning(f"LRC file not found: {lrc_path}")
                return []
            
            with open(lrc_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    match = re.match(r'\[(\d+):(\d+\.\d+)\](.*)', line)
                    if match:
                        minutes, seconds, text = match.groups()
                        timestamp = float(minutes) * 60 + float(seconds)
                        
                        lyrics.append({
                            'timestamp': timestamp,
                            'text': text.strip()
                        })
            
            logger.info(f"✓ Parsed {len(lyrics)} lyrics from {Path(lrc_path).name}")
            return lyrics if lyrics else []
            
        except Exception as e:
            logger.error(f"✗ Error parsing LRC: {e}")
            return []
    
    @staticmethod
    def get_lyrics_at_time(lyrics: List[Dict], timestamp: float) -> str:
        """Get lyric at specific timestamp"""
        if not lyrics:
            return ""
        
        for i, lyric in enumerate(lyrics):
            if i + 1 < len(lyrics):
                if lyrics[i]['timestamp'] <= timestamp < lyrics[i + 1]['timestamp']:
                    return lyric['text']
            elif lyric['timestamp'] <= timestamp:
                return lyric['text']
        
        return ""
