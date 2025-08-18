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
        
        # Save the dubbing script prompt for reference
        self.file_manager.save_prompt(
            prompt_content=prompt,
            prompt_type="dubbing_script",
            video_path=video_file.name if hasattr(video_file, 'name') else None,
            logger=logger
        )
        
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
        """Build enhanced prompt for Gemini API with natural speech replication focus."""
        # Check if we should use simplified prompt for better completeness
        use_simplified = config.get("SIMPLIFIED_PROMPT", False)
        
        if use_simplified:
            return self._build_simplified_prompt(config)
        
        return f"""
        You are an expert voice director and performance analyst creating a professional dubbing script.
        Your goal is NATURAL SPEECH REPLICATION - capturing not just words, but the complete human performance.

        ## CRITICAL REQUIREMENTS

        ### 0. COMPLETE VIDEO COVERAGE (ABSOLUTE REQUIREMENT)
        **MANDATORY**: Analyze the ENTIRE video from start to finish - do not stop early or skip any dialogue.
        - Process ALL spoken content throughout the complete video duration
        - Include EVERY piece of dialogue, even brief utterances or background conversations
        - Continue analysis until the very end of the video - do not truncate your response
        - If the video is long, prioritize completeness over detailed metadata for later segments
        - NEVER stop analyzing before reaching the end of the video content

        ### 1. TIMESTAMP ACCURACY & GAP ENFORCEMENT (MANDATORY)
        **CRITICAL**: Convert all times to precise seconds (1:30 = 90.0 seconds, 2:15 = 135.0 seconds)
        **MANDATORY GAP RULE**: Ensure 0.1-0.3 second minimum gap between ALL segments for natural speech flow
        - If segments would overlap, adjust timing to create natural spacing
        - Preserve authentic pauses within speech (breath pauses, thought pauses)
        - Validation rule: segment[i].end_time + 0.1 ≤ segment[i+1].start_time

        ### 2. SPEAKER IDENTIFICATION & CHARACTER CONTINUITY
        - Use audio characteristics AND visual cues for distinct speakers
        - Assign consistent labels: SPEAKER_1, SPEAKER_2, etc.
        - Track character voice patterns across all segments for continuity
        - Note any intentional voice changes (health, emotion, time passage)

        ### 3. CHARACTER CLASSIFICATION WITH VOICE QUALITY
        Classify each speaker with detailed vocal characteristics:
        - **MALE**: Adult male voice (note: deep, medium, high pitch range)
        - **FEMALE**: Adult female voice (note: pitch range, breathiness, resonance)
        - **CHILD**: Young voice (note: specific age impression if possible)
        - **ELDERLY**: Aged voice characteristics if applicable

        ### 4. PROSODIC ANALYSIS (Essential for Natural Speech)
        Listen for micro-performance details:
        
        **Intonation Patterns:**
        - **RISING**: Questions, uncertainty, lists (↗)
        - **FALLING**: Statements, completion (↘)
        - **RISE_FALL**: Emphasis, contrast (↗↘)
        - **FLAT**: Monotone, boredom, authority (→)
        
        **Voice Quality:**
        - **MODAL**: Normal, relaxed voice
        - **BREATHY**: Intimate, tired, sensual speech
        - **CREAKY**: Authority, low pitch, vocal fry
        - **TENSE**: Stress, anger, physical effort
        
        **Natural Pauses (Mark these precisely):**
        - **BREATH_PAUSE**: Natural breathing (0.1-0.3s)
        - **THOUGHT_PAUSE**: Processing, hesitation (0.3-0.8s)
        - **DRAMATIC_PAUSE**: Intentional silence (0.8s+)

        ### 5. ENHANCED EMOTION DETECTION
        Analyze emotional layers with intensity:
        
        **Primary Emotions:**
        - **HAPPY**: Light, upward inflection, energetic
        - **SAD**: Heavy, downward inflection, slower
        - **ANGRY**: Sharp, tense, louder volume
        - **FEARFUL**: Trembling, uncertain, hesitant
        - **SURPRISED**: Sudden pitch changes, exclamations
        - **NEUTRAL**: Calm, even tone, conversational
        
        **Secondary Emotions (be specific):**
        - **EXCITED, DISAPPOINTED, FRUSTRATED, CONFUSED, CONFIDENT, NERVOUS, SARCASTIC, LOVING**
        
        **Emotional Intensity:**
        - **MILD**: Subtle emotional coloring
        - **MODERATE**: Clear emotional expression
        - **INTENSE**: Strong emotional dominance

        ### 6. ADVANCED DELIVERY STYLE ANALYSIS
        Identify nuanced performance characteristics:
        
        **Basic Styles:**
        - **NORMAL, SHOUTING, WHISPERING, CRYING, PLEADING, LAUGHING**
        
        **Conversational Styles:**
        - **STORYTELLING**: Narrative, engaging rhythm
        - **EXPLAINING**: Clear, methodical delivery
        - **ARGUING**: Confrontational, sharp edges
        - **FLIRTING**: Playful, melodic variations
        - **COMMANDING**: Authoritative, direct
        
        **Emotional Delivery Variants:**
        - **SUPPRESSED_ANGER**: Controlled but tense
        - **FORCED_HAPPINESS**: Artificial cheerfulness
        - **NERVOUS_LAUGHTER**: Anxiety masking
        - **QUIET_DESPERATION**: Subtle emotional plea

        ### 7. CONTEXTUAL PERFORMANCE ANALYSIS
        Consider broader scene dynamics:
        
        **Relationship Context:**
        - **DOMINANT**: Speaker has authority/power
        - **SUBMISSIVE**: Deferential, respectful
        - **EQUAL**: Peer-level conversation
        - **INTIMATE**: Close relationship, private moment
        
        **Scene Energy:**
        - **BUILDUP**: Tension increasing
        - **CLIMAX**: Peak emotional moment
        - **RESOLUTION**: Tension releasing
        - **CASUAL**: Low-stakes interaction

        ### 8. PACE & RHYTHM ANALYSIS
        Measure speech patterns precisely:
        - **VERY_SLOW**: Deliberate, dramatic emphasis
        - **SLOW**: Measured, thoughtful delivery
        - **NORMAL**: Standard conversational rhythm
        - **FAST**: Quick, energetic, excited
        - **VERY_FAST**: Rapid, rushed, overwhelming
        - **IRREGULAR**: Varied pace within segment

        ### 9. TRANSCRIPTION & CULTURAL TRANSLATION
        - Provide accurate timestamped '{config["INPUT_LANGUAGE"]}' transcription
        - Create natural '{config["OUTPUT_LANGUAGE"]}' translation preserving:
          * Emotional authenticity
          * Cultural appropriateness
          * Colloquial speech patterns
          * Character voice consistency
        
        {self._build_verbosity_instructions(config) if config.get("ADJUST_VERBOSITY") else ""}

        ## ENHANCED OUTPUT FORMAT
        Return ONLY a valid JSON array. Each segment must include:

        {{
          "start_time": float,
          "end_time": float,
          "speaker_label": "string",
          "character_type": "MALE|FEMALE|CHILD|ELDERLY",
          "emotion": "emotion_name",
          "emotion_intensity": "MILD|MODERATE|INTENSE",
          "delivery_style": "delivery_style_name",
          "intonation_pattern": "RISING|FALLING|RISE_FALL|FLAT",
          "voice_quality": "MODAL|BREATHY|CREAKY|TENSE",
          "pace": "VERY_SLOW|SLOW|NORMAL|FAST|VERY_FAST|IRREGULAR",
          "relationship_context": "DOMINANT|SUBMISSIVE|EQUAL|INTIMATE",
          "scene_energy": "BUILDUP|CLIMAX|RESOLUTION|CASUAL",
          "natural_pauses": ["BREATH_PAUSE", "THOUGHT_PAUSE", "DRAMATIC_PAUSE"],
          "original_transcript": "string",
          "{config["OUTPUT_LANGUAGE"]}_translation": "string",
          "prosodic_notes": "Additional performance guidance for natural delivery"
        }}

        ## FINAL REMINDER: COMPLETENESS IS CRITICAL
        - Ensure you have processed the ENTIRE video duration
        - Do NOT stop your analysis prematurely 
        - Include ALL dialogue segments from start to finish
        - If response length is a concern, reduce metadata detail but maintain complete segment coverage
        
        REMEMBER: Your analysis determines how natural the final dubbing sounds. Focus on capturing the human performance, not just the words. MOST IMPORTANTLY: Cover the entire video completely.
        """
    
    def _build_simplified_prompt(self, config):
        """Build simplified prompt prioritizing completeness over detailed metadata."""
        # Build verbosity adjustment instructions
        verbosity_instructions = self._build_verbosity_instructions(config) if config.get("ADJUST_VERBOSITY") else ""
        
        return f"""
        You are creating a complete dubbing script for this video. 
        
        CRITICAL: Analyze the ENTIRE video from start to finish. Do not stop early or skip any dialogue.
        
        Requirements:
        1. COMPLETE COVERAGE: Process ALL dialogue in the video from beginning to end
        2. ACCURATE TIMESTAMPS: Convert times to seconds (1:30 = 90.0, 2:15 = 135.0)
        3. SPEAKER IDENTIFICATION: Use SPEAKER_1, SPEAKER_2, etc. consistently
        4. BASIC CHARACTER TYPES: MALE, FEMALE, CHILD, or ELDERLY
        5. TRANSLATIONS: Provide natural {config["OUTPUT_LANGUAGE"]} translation
        
        {verbosity_instructions}
        
        Return ONLY a JSON array with the SAME format as the full prompt:
        
        {{
          "start_time": float,
          "end_time": float,
          "speaker_label": "string",
          "character_type": "MALE|FEMALE|CHILD|ELDERLY",
          "emotion": "emotion_name",
          "emotion_intensity": "MILD|MODERATE|INTENSE",
          "delivery_style": "delivery_style_name",
          "intonation_pattern": "RISING|FALLING|RISE_FALL|FLAT",
          "voice_quality": "MODAL|BREATHY|CREAKY|TENSE",
          "pace": "VERY_SLOW|SLOW|NORMAL|FAST|VERY_FAST|IRREGULAR",
          "relationship_context": "DOMINANT|SUBMISSIVE|EQUAL|INTIMATE",
          "scene_energy": "BUILDUP|CLIMAX|RESOLUTION|CASUAL",
          "natural_pauses": ["BREATH_PAUSE", "THOUGHT_PAUSE", "DRAMATIC_PAUSE"],
          "original_transcript": "string",
          "{config["OUTPUT_LANGUAGE"]}_translation": "string",
          "prosodic_notes": "Additional performance guidance for natural delivery"
        }}
        
        ABSOLUTE PRIORITY: Include EVERY piece of dialogue in the video. Completeness is more important than detailed analysis.
        Use basic values for metadata fields (e.g., "NEUTRAL" for emotion, "NORMAL" for most fields) to focus on completeness.
        """
    
    def _build_verbosity_instructions(self, config):
        """Build verbosity adjustment instructions for cross-language intent translation."""
        input_lang = config["INPUT_LANGUAGE"].lower()
        output_lang = config["OUTPUT_LANGUAGE"].lower()
        
        return f"""
        
        ### VERBOSITY ADJUSTMENT FOR CROSS-LANGUAGE INTENT
        **CRITICAL**: Focus on communicating the INTENT and MEANING rather than literal translation.
        Adjust verbosity between {config["INPUT_LANGUAGE"]} and {config["OUTPUT_LANGUAGE"]} for natural dubbing:
        
        **Language-Specific Adaptations:**
        
        **If translating TO more concise languages** (English, German, Japanese):
        - Condense verbose expressions while preserving emotional impact
        - Convert lengthy explanations to punchy, direct statements
        - Maintain the speaker's intended emphasis and tone
        - Example: Long descriptive phrase → Impactful short phrase with same emotional weight
        
        **If translating TO more expressive languages** (Spanish, Italian, Hindi, Arabic):
        - Expand terse statements to match natural expressiveness of target language
        - Add culturally appropriate emotional markers and emphasis
        - Include natural hesitations, confirmations, and conversational flow
        - Example: "Yes" → "Yes, absolutely" or equivalent cultural expression
        
        **Cross-Language Intent Preservation:**
        - Maintain the speaker's confidence level (assertive vs uncertain)
        - Preserve relationship dynamics (formal vs casual, respectful vs familiar)
        - Keep emotional intensity equivalent across language patterns
        - Adjust sentence structure to match target language's natural rhythm
        
        **Dubbing Quality Focus:**
        - Ensure translated text feels natural when spoken aloud
        - Consider mouth movement compatibility where possible
        - Prioritize speakability over literal accuracy
        - Match the energy and flow of original speech patterns
        
        **Intent Categories to Preserve:**
        - **COMMANDING**: Authority and directness
        - **PERSUADING**: Convincing and influential tone
        - **QUESTIONING**: Genuine inquiry vs rhetorical questioning
        - **EXPLAINING**: Informative vs condescending delivery
        - **EMOTIONAL**: Joy, anger, sadness intensity levels
        - **SOCIAL**: Politeness, intimacy, professional distance
        """
    
    def _parse_gemini_response(self, response, logger):
        """Parse and validate Gemini API response with enhanced timestamp validation."""
        try:
            json_text = response.text.strip().lstrip("```json").rstrip("```")
            
            # Log response characteristics for debugging
            logger(f"📊 Gemini response analysis:")
            logger(f"   📏 Response length: {len(response.text)} characters")
            logger(f"   📝 JSON content length: {len(json_text)} characters")
            
            # Check for potential truncation indicators
            if json_text.endswith('...') or not json_text.endswith('}]'):
                logger("⚠️ WARNING: Response may be truncated - doesn't end with proper JSON closing")
            
            if len(response.text) > 30000:
                logger("⚠️ WARNING: Very long response - may hit API limits")
                
            dubbing_script = json.loads(json_text)
            dubbing_script.sort(key=lambda x: x['start_time'])
            
            # Log segment timing analysis
            if dubbing_script:
                first_segment = dubbing_script[0]['start_time']
                last_segment = dubbing_script[-1]['end_time']
                logger(f"   🕐 Segment range: {first_segment:.1f}s to {last_segment:.1f}s ({last_segment - first_segment:.1f}s total)")
                
            # Validate and fix timestamp overlaps
            validated_script = self._validate_and_fix_timestamps(dubbing_script, logger)
            
            logger("✅ Successfully received and parsed the dubbing script from Gemini.")
            logger(f"📊 Generated {len(validated_script)} dialogue segments with validated timestamps.")
            return validated_script
            
        except (json.JSONDecodeError, IndexError, AttributeError) as e:
            logger(f"❌ Failed to parse JSON from Gemini's response: {e}")
            logger(f"RAW RESPONSE:\n{response.text}")
            return None
    
    def _validate_and_fix_timestamps(self, dubbing_script, logger):
        """Validate timestamps and fix overlaps with minimum gap enforcement."""
        if not dubbing_script:
            return dubbing_script
        
        MIN_GAP = 0.1  # Minimum 0.1 second gap between segments
        fixes_applied = 0
        
        # Ensure segments are sorted by start time
        dubbing_script.sort(key=lambda x: x['start_time'])
        
        for i in range(len(dubbing_script) - 1):
            current_segment = dubbing_script[i]
            next_segment = dubbing_script[i + 1]
            
            # Check for overlap or insufficient gap
            current_end = current_segment['end_time']
            next_start = next_segment['start_time']
            
            if current_end + MIN_GAP > next_start:
                # Fix overlap by adjusting current segment's end time
                # Leave minimum gap for natural speech flow
                new_end_time = next_start - MIN_GAP
                
                # Ensure the segment doesn't become negative duration
                if new_end_time > current_segment['start_time']:
                    current_segment['end_time'] = round(new_end_time, 2)
                    fixes_applied += 1
                    
                    logger(f"   🔧 Fixed overlap: Segment {i+1} end time adjusted to {new_end_time:.2f}s")
                else:
                    # If adjustment would make negative duration, adjust next segment instead
                    next_segment['start_time'] = current_end + MIN_GAP
                    fixes_applied += 1
                    
                    logger(f"   🔧 Fixed overlap: Segment {i+2} start time adjusted to {current_end + MIN_GAP:.2f}s")
        
        if fixes_applied > 0:
            logger(f"✅ Applied {fixes_applied} timestamp fixes to prevent audio overlaps.")
        else:
            logger("✅ All timestamps validated - no overlaps detected.")
        
        return dubbing_script