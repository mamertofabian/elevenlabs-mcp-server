import os
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional
from pydub import AudioSegment
import io
from datetime import datetime
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

class ElevenLabsAPI:
    # Add model list as class constant
    MODELS = {
        "eleven_multilingual_v2": {"description": "Our most lifelike model with rich emotional expression", "languages": "32",
                                   "supports_stitching": True, "supports_style": True, "wait_time": 0.1},
        "eleven_flash_v2_5": {"description": "Ultra-fast model optimized for real-time use (~75ms†)", "languages": "32",
                              "supports_stitching": False, "supports_style": False, "wait_time": 0.1},
        "eleven_flash_v2": {"description": "Ultra-fast model optimized for real-time use (~75ms†)", "languages": "English",
                             "supports_stitching": False, "supports_style": False, "wait_time": 0.1}
    }

    def __init__(self):
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID") or "iEw1wkYocsNy7I7pteSN"
        self.model_id = os.getenv("ELEVENLABS_MODEL_ID") or "eleven_multilingual_v2"
        # Add validation for model_id
        if self.model_id not in self.MODELS:
            raise ValueError(f"Invalid model_id: {self.model_id}. Must be one of {list(self.MODELS.keys())}")
        self.stability = float(os.getenv("ELEVENLABS_STABILITY", "0.5"))
        self.similarity_boost = float(os.getenv("ELEVENLABS_SIMILARITY_BOOST", "0.75"))
        self.style = float(os.getenv("ELEVENLABS_STYLE", "0.1"))
        self.base_url = "https://api.elevenlabs.io/v1"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_audio_segment(self, text: str, voice_id: str, output_file: Optional[str] = None,
                      previous_text: Optional[str] = None, next_text: Optional[str] = None,
                      previous_request_ids: Optional[List[str]] = None, debug_info: Optional[List[str]] = None) -> tuple[bytes, str]:
        """Generate audio using specified voice with context conditioning"""
        headers = {
            "Accept": "application/json",
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        data = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": self.stability,
                "similarity_boost": self.similarity_boost
            }
        }

        if self.MODELS[self.model_id]["supports_style"]:
            data["style"] = self.style

        # Add context conditioning if model supports it
        if self.MODELS[self.model_id]["supports_stitching"]:
            if previous_text is not None:
                data["previous_text"] = previous_text
            if next_text is not None:
                data["next_text"] = next_text
            if previous_request_ids:
                data["previous_request_ids"] = previous_request_ids[-3:]  # Maximum of 3 previous IDs
        
        response = requests.post(
            f"{self.base_url}/text-to-speech/{voice_id}",
            json=data,
            headers=headers
        )
        
        if response.status_code == 200:
            if output_file:
                with open(output_file, 'wb') as f:
                    f.write(response.content)
            return response.content, response.headers["request-id"]
        else:
            debug_info.append(response.text)
            error_message = f"Failed to generate audio: {response.text} \n\n{debug_info} \n\n{data}"
            raise Exception(error_message)

    def generate_full_audio(self, script_parts: List[Dict], output_dir: Path) -> tuple[str, List[str]]:
        """Generate audio for multiple parts using request stitching. Returns tuple of (output_file_path, debug_info)"""
        # Create output directory if it doesn't exist
        output_dir.mkdir(exist_ok=True)
        
        # Final output file path with unique file name
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_file = output_dir / f"full_audio_{timestamp}.mp3"
        
        debug_info = []
        debug_info.append("ElevenLabsAPI - Starting generate_full_audio")
        debug_info.append(f"Input script_parts: {script_parts}")
        
        # Initialize segments list and request IDs tracking
        segments = []
        previous_request_ids = []
        failed_parts = []
        
        debug_info.append("Processing all_texts")
        all_texts = []
        for part in script_parts:
            debug_info.append(f"Processing text from part: {part}")
            text = str(part.get('text', ''))
            debug_info.append(f"Extracted text: {text}")
            all_texts.append(text)
        debug_info.append(f"Final all_texts: {all_texts}")
        
        for i, part in enumerate(script_parts):
            debug_info.append(f"Processing part {i}: {part}")
            part_voice_id = part.get('voice_id')
            if not part_voice_id:
                part_voice_id = self.voice_id
            text = str(part.get('text', ''))
            if not text:
                continue
                
            debug_info.append(f"Using voice ID: {part_voice_id}")
            
            # Determine previous and next text for context
            is_first = i == 0
            is_last = i == len(script_parts) - 1
            
            previous_text = None if is_first else " ".join(all_texts[:i])
            next_text = None if is_last else " ".join(all_texts[i + 1:])
            
            try:
                # Generate audio with context conditioning
                audio_content, request_id = self.generate_audio_segment(
                    text=text,
                    voice_id=part_voice_id,
                    previous_text=previous_text,
                    next_text=next_text,
                    previous_request_ids=previous_request_ids,
                    debug_info=debug_info
                )
                
                debug_info.append(f"Successfully generated audio for part {i}")
                
                # Add request ID to history
                previous_request_ids.append(request_id)
                
                # Convert audio content to AudioSegment and add to segments
                audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_content))
                segments.append(audio_segment)

                # Wait for the specified wait_time
                time.sleep(self.MODELS[self.model_id]["wait_time"])
            except Exception as e:
                debug_info.append(f"Error generating audio: {e}")
                failed_parts.append(part)
                continue
        
        # Combine all segments
        if segments:
            final_audio = segments[0]
            for segment in segments[1:]:
                final_audio = final_audio + segment
            
            # Export combined audio
            final_audio.export(output_file, format="mp3")

            if failed_parts:
                debug_info.append(f"Failed parts: {failed_parts}")
            else:
                debug_info.append("All parts generated successfully")
            
            debug_info.append(f"Model: {self.model_id}")
            
            return str(output_file), debug_info
        else:
            error_msg = "\n".join([
                "No audio segments were generated. Debug info:",
                *debug_info
            ])
            raise Exception(error_msg)
