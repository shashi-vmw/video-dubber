import os
import time
import json
import google.genai as genai
from file_manager import FileManager


class DubbingScriptGenerator:
    """Handles interaction with Gemini API for dubbing script generation."""
    
    def __init__(self, file_manager=None):
        self.file_manager = file_manager or FileManager()
    
    def generate_dubbing_script(self, video_path, config, logger):
        """
        Uploads a video to the Gemini API and generates a timestamped,
        translated, and character-identified dubbing script.
        """
        # If no API key is provided, we cannot generate a script.
        if not config.get("GEMINI_API_KEY") and not config.get("USE_VERTEX_AI"):
            if config.get("EXTRACTION_ONLY"):
                logger("⚠️ Skipping script generation as no API key was provided.")
                return None
            else:
                logger("❌ Cannot generate script without a Gemini API Key or Vertex AI configuration.")
                return None

        # Check if we should reuse an existing script
        if config.get("REUSE_PATH"):
            # Note: Reusing a script from a path is handled by the file manager's directory setup.
            # Here, we check for a local script file to reuse if the flag was more general.
            # This logic can be refined if needed.
            script_path = self.file_manager.get_dubbing_script_path(video_path)
            if os.path.exists(script_path):
                logger("✅ Reusing existing dubbing script.")
                try:
                    with open(script_path, 'r') as f:
                        return json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    logger(f"⚠️ Could not read existing script file: {e}. Regenerating...")
        
        try:
            # Authenticate with Gemini API
            client = self._authenticate_gemini(config, logger)
            if not client:
                return None
            
            # Upload video to Gemini
            video_file = self._upload_video_to_gemini(client, video_path, logger)
            if not video_file:
                return None
            
            # Generate dubbing script
            dubbing_script = self._generate_script_with_gemini(client, video_file, config, logger)
            
            # Clean up uploaded file
            client.files.delete(name=video_file.name)
            logger("🧹 Cleaned up uploaded file on server.")
            
            if dubbing_script:
                # Save dubbing script to working directory
                self.file_manager.save_dubbing_script(video_path, dubbing_script, logger)
            
            return dubbing_script
            
        except Exception as e:
            logger(f"❌ Unexpected error in dubbing script generation: {str(e)}")
            return None
    
    def assign_voices_to_speakers(self, transcript_data):
        """Assign specific voices to speakers based on character type."""
        # Voice configuration
        MALE_VOICE_LIST = ['Puck', 'Orus', 'Enceladus']
        FEMALE_VOICE_LIST = ['Kore', 'Zephyr', 'Leda', 'Sulafat', 'Aoede']
        CHILD_VOICE_LIST = ['Leda', 'Kore']
        FALLBACK_VOICE = 'Leda'
        
        speaker_info = {}
        for item in transcript_data:
            speaker_label = item['speaker_label']
            if speaker_label not in speaker_info:
                speaker_info[speaker_label] = item['character_type']

        voice_indices = {'MALE': 0, 'FEMALE': 0, 'CHILD': 0}
        speaker_voice_array = []
        sorted_speakers = sorted(speaker_info.keys())

        for speaker in sorted_speakers:
            char_type = speaker_info[speaker]
            selected_voice = ""
            
            if char_type == 'FEMALE' and FEMALE_VOICE_LIST:
                index = voice_indices['FEMALE'] % len(FEMALE_VOICE_LIST)
                selected_voice = FEMALE_VOICE_LIST[index]
                voice_indices['FEMALE'] += 1
            elif char_type == 'MALE' and MALE_VOICE_LIST:
                index = voice_indices['MALE'] % len(MALE_VOICE_LIST)
                selected_voice = MALE_VOICE_LIST[index]
                voice_indices['MALE'] += 1
            elif char_type == 'CHILD' and CHILD_VOICE_LIST:
                index = voice_indices['CHILD'] % len(CHILD_VOICE_LIST)
                selected_voice = CHILD_VOICE_LIST[index]
                voice_indices['CHILD'] += 1
            else:
                selected_voice = FALLBACK_VOICE
                
            speaker_voice_array.append({
                "speaker_label": speaker, 
                "character_type": char_type, 
                "selected_voice": selected_voice
            })
            
        return speaker_voice_array
    
    def _authenticate_gemini(self, config, logger):
        """Authenticate with Gemini API."""
        try:
            if config.get("USE_VERTEX_AI"):
                logger(f"Authenticating with Vertex AI (Project: {config['PROJECT_ID']}, Location: {config['LOCATION']})")
                return genai.Client(project=config["PROJECT_ID"], location=config["LOCATION"])
            else:
                logger("Authenticating with Google API Key")
                return genai.Client(api_key=config["GEMINI_API_KEY"])
        except Exception as e:
            logger(f"❌ Authentication failed: {e}")
            return None
    
    def _upload_video_to_gemini(self, client, video_path, logger):
        """Upload video file to Gemini API."""
        logger(f"Uploading video '{os.path.basename(video_path)}' to the Gemini API...")
        
        try:
            video_file = client.files.upload(file=video_path)
            
            processing_message = "Video is processing..."
            logger(processing_message)
            
            # Wait for video processing to complete
            while video_file.state.name == "PROCESSING":
                time.sleep(10)
                video_file = client.files.get(name=video_file.name)

            if video_file.state.name == "FAILED":
                logger(f"❌ Video processing failed: {video_file.state}")
                return None

            logger(f"✅ Video uploaded and processed successfully: {video_file.name}")
            return video_file
            
        except Exception as e:
            logger(f"❌ File upload failed: {e}")
            return None
    
    def _generate_script_with_gemini(self, client, video_file, config, logger):
        """Generate dubbing script using Gemini API."""
        prompt = self._build_dubbing_prompt(config)
        
        logger(f"🤖 Sending request to {config['MODEL_NAME']} for analysis...")
        
        try:
            response = client.models.generate_content(
                model=config['MODEL_NAME'],
                contents=[video_file, "\n\n", prompt]
            )
            
            return self._parse_gemini_response(response, logger)
            
        except Exception as e:
            logger(f"❌ Error calling Gemini API: {e}")
            return None
    
    def _build_dubbing_prompt(self, config):
        """Build the prompt for Gemini API with optimized emotion and delivery style detection."""
        return f"""
        You are an expert voice director and video producer creating a professional dubbing script.
        Analyze the video file's audio and visual elements with precision to capture complete performance nuances.

        ## ANALYSIS FRAMEWORK

        ### 1. SPEAKER IDENTIFICATION
        - Use both audio characteristics and visual cues to identify distinct speakers
        - Assign unique labels: SPEAKER_1, SPEAKER_2, etc.
        - Consider lip-sync patterns and visual presence

        ### 2. CHARACTER CLASSIFICATION  
        Classify each speaker as:
        - **MALE**: Adult male voice characteristics
        - **FEMALE**: Adult female voice characteristics  
        - **CHILD**: Young voice, higher pitch, immature speech patterns

        ### 3. EMOTION DETECTION (Primary Focus)
        Listen carefully to vocal tone, pitch variations, and speech patterns. Classify the dominant emotion:
        
        **Examples for reference:**
        - **HAPPY**: Upward inflection, lighter tone, faster pace, laughter
        - **SAD**: Downward inflection, slower pace, quieter volume, sighs
        - **ANGRY**: Sharp tone, louder volume, clipped speech, tension
        - **SURPRISED**: Sudden pitch changes, exclamations, interrupted speech
        - **FEARFUL**: Trembling voice, whispered tone, hesitation
        - **NEUTRAL**: Even tone, steady pace, conversational delivery

        ### 4. DELIVERY STYLE ANALYSIS (Enhanced Detection)
        Identify vocal delivery characteristics:
        
        **Style Categories:**
        - **NORMAL**: Standard conversational tone
        - **SHOUTING**: Loud volume, projected voice, emphasis
        - **WHISPERING**: Very quiet, intimate, secretive tone
        - **CRYING**: Voice breaking, sobbing, emotional distress
        - **PLEADING**: Urgent, begging tone, desperation
        - **LAUGHING**: Joyful, with laughter interrupting speech

        ### 5. PACE CLASSIFICATION
        Measure speech rhythm and timing:
        - **VERY SLOW**: Deliberate, drawn-out words
        - **SLOW**: Measured, careful speech
        - **NORMAL**: Standard conversational pace
        - **FAST**: Quick, energetic delivery
        - **VERY FAST**: Rapid, excited, or rushed speech

        ### 6. TRANSCRIPTION & TRANSLATION
        - Provide accurate timestamped '{config["INPUT_LANGUAGE"]}' transcription
        - Create natural '{config["OUTPUT_LANGUAGE"]}' translation maintaining emotional context
        - Preserve cultural nuances and idiomatic expressions

        ### 7. TIMESTAMP ACCURACY
        **CRITICAL**: Convert all times to seconds only.
        - Format 1:15 = 75 seconds (not 115)
        - Format 2:30 = 150 seconds (not 230)

        ## OUTPUT FORMAT
        Return ONLY a valid JSON array. Each dialogue segment must include:

        {{
          "start_time": float,
          "end_time": float,
          "speaker_label": "string",
          "character_type": "MALE|FEMALE|CHILD",
          "emotion": "HAPPY|SAD|ANGRY|SURPRISED|FEARFUL|NEUTRAL",
          "delivery_style": "NORMAL|SHOUTING|WHISPERING|CRYING|PLEADING|LAUGHING",
          "original_transcript": "string",
          "{config["OUTPUT_LANGUAGE"]}_translation": "string",
          "pace": "VERY_SLOW|SLOW|NORMAL|FAST|VERY_FAST"
        }}

        Focus on audio-visual correlation for maximum accuracy in emotion and delivery style detection.
        """
    
    def _parse_gemini_response(self, response, logger):
        """Parse and validate Gemini API response."""
        try:
            json_text = response.text.strip().lstrip("```json").rstrip("```")
            dubbing_script = json.loads(json_text)
            dubbing_script.sort(key=lambda x: x['start_time'])
            
            logger("✅ Successfully received and parsed the dubbing script from Gemini.")
            return dubbing_script
            
        except (json.JSONDecodeError, IndexError, AttributeError) as e:
            logger(f"❌ Failed to parse JSON from Gemini's response: {e}")
            logger(f"RAW RESPONSE:\n{response.text}")
            return None