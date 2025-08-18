import os
import time
import wave
import subprocess
import google.genai as genai
from google.genai import types
from moviepy import VideoFileClip, AudioFileClip
from pydub import AudioSegment
import pyrubberband as rb
import soundfile as sf


class AudioProcessor:
    """Handles audio extraction, processing, and synthesis for video dubbing."""

    def extract_audio(self, video_path, audio_path, logger):
        """
        Extracts and merges all audio streams from a video file into a single stereo WAV file
        using a direct ffmpeg call. This is robust for videos with multiple mono tracks.
        """
        logger(f"🎥 Extracting and merging all audio streams from '{os.path.basename(video_path)}'...")
        
        try:
            # First, probe the file to count the number of audio streams
            probe_command = [
                "ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", 
                "stream=index", "-of", "csv=p=0", video_path
            ]
            probe_result = subprocess.run(probe_command, check=True, capture_output=True, text=True)
            num_audio_streams = len(probe_result.stdout.strip().split('\n'))
            
            if num_audio_streams == 0:
                logger("❌ No audio streams found in the video file.")
                return None
            
            logger(f"   Found {num_audio_streams} audio stream(s).")

            # Build the ffmpeg command to merge all audio streams
            command = [
                "ffmpeg", "-i", video_path, "-y",
                "-filter_complex", f"[0:a]amerge=inputs={num_audio_streams}[a]",
                "-map", "[a]",
                "-acodec", "pcm_s16le",
                "-ar", "44100",
                "-ac", "2",
                audio_path
            ]
            
            logger("   Running ffmpeg to merge and extract audio...")
            result = subprocess.run(command, check=True, capture_output=True, text=True)

            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                logger("❌ ffmpeg command ran, but the output audio file is missing or empty.")
                logger(f"   💬 ffmpeg stderr: {result.stderr}")
                return None

            logger(f"✅ Audio extracted successfully to '{os.path.basename(audio_path)}'")
            return audio_path
            
        except FileNotFoundError:
            logger("❌ ffmpeg or ffprobe command not found. Please ensure FFmpeg is installed and in your system's PATH.")
            return None
        except subprocess.CalledProcessError as e:
            logger(f"❌ Error during audio extraction/probing: Command failed with exit code {e.returncode}")
            logger(f"   💬 Command: {' '.join(e.cmd)}")
            logger(f"   💬 Stderr: {e.stderr}")
            return None
        except Exception as e:
            logger(f"❌ An unexpected error occurred during audio extraction: {e}")
            return None

    def separate_background_music(self, audio_path, output_dir, logger):
        """Separate background music from vocals using Demucs."""
        logger(
            "🎶 Separating background music with Demucs... (This might take a while)"
        )
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)

            command = [
                "python3",
                "-m",
                "demucs.separate",
                "-n",
                "htdemucs",
                "-o",
                str(output_dir),
                "--two-stems",
                "vocals",
                str(audio_path),
            ]

            logger("   🔧 Running Demucs separation...")
            result = subprocess.run(command, check=True, capture_output=True, text=True)

            audio_filename = os.path.splitext(os.path.basename(audio_path))[0]
            model_name = "htdemucs"
            background_path = os.path.join(
                output_dir, model_name, audio_filename, "no_vocals.wav"
            )

            if os.path.exists(background_path):
                # Check if background file has actual content
                background_audio = AudioSegment.from_wav(background_path)
                if len(background_audio) > 0:
                    logger("✅ Background music separated successfully.")
                    logger(
                        f"   📊 Background track duration: {len(background_audio)/1000:.1f} seconds"
                    )
                    return background_path
                else:
                    logger("⚠️ Background track is empty - using original audio instead")
                    return None
            else:
                logger("❌ Demucs did not produce the background music file.")
                logger(f"   💬 Expected path: {background_path}")
                if result.stderr:
                    logger(f"   💬 Demucs stderr: {result.stderr}")
                if result.stdout:
                    logger(f"   💬 Demucs stdout: {result.stdout}")
                return None
        except subprocess.CalledProcessError as e:
            logger(f"❌ Demucs command failed: {e}")
            logger(f"   💬 Command: {' '.join(command)}")
            logger(f"   💬 Return code: {e.returncode}")
            if e.stderr:
                logger(f"   💬 stderr: {e.stderr}")
            return None
        except Exception as e:
            logger(f"❌ An error occurred during audio separation: {e}")
            return None

    def synthesize_speech_with_gemini(self, text, segment_details, config, logger):
        """Synthesize speech using Gemini TTS API with retry mechanism."""
        logger(
            f"   Synthesizing: '{text[:40]}...' for {segment_details['speaker_label']}"
        )

        # Retry configuration
        max_retries = config.get("TTS_MAX_RETRIES", 3)
        base_delay = config.get("TTS_RETRY_DELAY", 2)  # seconds
        
        for attempt in range(max_retries + 1):
            try:
                # Authenticate with Gemini API
                client = self._authenticate_gemini(config, logger)
                if not client:
                    return None

                # Build voice description and prompt
                full_prompt = self._build_tts_prompt(text, segment_details, config)

                # Generate speech with retry logging
                if attempt > 0:
                    logger(f"   🔄 Retry attempt {attempt}/{max_retries} for TTS synthesis")

                response = client.models.generate_content(
                    model=config["TTS_MODEL"],
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=segment_details["selected_voice"],
                                )
                            )
                        ),
                    ),
                )

                # Extract audio data from response
                result = self._extract_audio_from_response(
                    response, segment_details["output_path"], logger
                )
                
                # If successful, return result
                if result:
                    if attempt > 0:
                        logger(f"   ✅ TTS synthesis succeeded on retry {attempt}")
                    return result
                
                # If extraction failed but no exception, treat as retryable failure
                if attempt < max_retries:
                    delay = self._calculate_retry_delay(attempt, base_delay)
                    logger(f"   ⚠️ Empty TTS response, retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                else:
                    logger(f"   ❌ TTS failed after {max_retries} retries: Empty response")
                    return None

            except Exception as e:
                error_msg = str(e).lower()
                
                # Check if error is retryable
                if self._is_retryable_error(error_msg):
                    if attempt < max_retries:
                        delay = self._calculate_retry_delay(attempt, base_delay)
                        logger(f"   ⚠️ Retryable TTS error: {e}")
                        logger(f"   🔄 Retrying in {delay}s (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(delay)
                        continue
                    else:
                        logger(f"   ❌ TTS failed after {max_retries} retries: {e}")
                        return None
                else:
                    # Non-retryable error, fail immediately
                    logger(f"   ❌ Non-retryable TTS error: {e}")
                    return None
        
        return None

    def _is_retryable_error(self, error_msg):
        """Determine if an error is retryable."""
        retryable_indicators = [
            'timeout',
            'connection',
            'network',
            'rate limit',
            'quota',
            'temporary',
            'service unavailable',
            'internal error',
            'deadline exceeded',
            'throttled',
            '429',  # Rate limit HTTP code
            '500',  # Internal server error
            '502',  # Bad gateway
            '503',  # Service unavailable
            '504',  # Gateway timeout
        ]
        
        return any(indicator in error_msg for indicator in retryable_indicators)

    def _calculate_retry_delay(self, attempt, base_delay):
        """Calculate exponential backoff delay with jitter."""
        import random
        
        # Exponential backoff: base_delay * 2^attempt
        delay = base_delay * (2 ** attempt)
        
        # Add jitter (±25% randomization) to avoid thundering herd
        jitter = delay * 0.25 * random.random()
        final_delay = delay + jitter
        
        # Cap maximum delay at 30 seconds
        return min(final_delay, 30.0)

    def merge_audio_with_video(self, video_path, audio_path, output_path, logger):
        """Merge final audio with video."""
        logger(f"🎬 Merging final audio with video...")
        logger(f"   📁 Video: {os.path.basename(video_path)}")
        logger(f"   🎵 Audio: {os.path.basename(audio_path)}")

        try:
            # Check if audio file exists and has content
            if not os.path.exists(audio_path):
                logger(f"❌ Audio file not found: {audio_path}")
                return None

            # Load and verify audio content
            from pydub import AudioSegment

            test_audio = AudioSegment.from_wav(audio_path)
            logger(
                f"   📊 Audio file loaded - Duration: {len(test_audio)/1000:.1f}s, Max amplitude: {test_audio.max_possible_amplitude}"
            )

            with VideoFileClip(video_path) as video_clip, AudioFileClip(
                audio_path
            ) as audio_clip:
                logger(f"   📹 Original video duration: {video_clip.duration:.1f}s")
                logger(f"   🎵 Audio clip duration: {audio_clip.duration:.1f}s")

                # Replace video's audio with our mixed audio
                video_clip.audio = audio_clip

                # Write final video with high quality audio settings
                video_clip.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec="aac",
                    temp_audiofile="temp-audio.m4a",
                    remove_temp=True,
                    logger=None,
                )

            logger(
                f"🎉 Final video saved successfully to '{os.path.basename(output_path)}'"
            )
            logger(
                f"   📊 Final video size: {os.path.getsize(output_path) / (1024**2):.1f} MB"
            )
            return output_path
        except Exception as e:
            logger(f"❌ An error occurred during video merging: {e}")
            logger(f"   💬 Video path: {video_path}")
            logger(f"   💬 Audio path: {audio_path}")
            logger(f"   💬 Output path: {output_path}")
            return None

    def process_audio_segments(
        self,
        dubbing_script,
        speaker_assignments,
        config,
        output_dir,
        background_music,
        logger,
        file_manager=None,
    ):
        """Process all audio segments for dubbing."""
        final_vocal_track = AudioSegment.silent(duration=len(background_music))
        output_lang_key = f"{config['OUTPUT_LANGUAGE']}_translation"
        total_lines = len(dubbing_script)
        successful_syntheses = 0

        FALLBACK_VOICE = "Leda"
        
        logger(f"🔄 Processing {total_lines} segments for audio synthesis...")
        logger(f"   📊 Background music duration: {len(background_music)/1000:.1f}s")

        for i, segment in enumerate(dubbing_script):
            logger(f"   🎯 Processing segment {i+1}/{total_lines}: {segment.get('speaker_label', 'UNKNOWN')}")
            
            output_text = segment.get(output_lang_key, "...")
            start_time_ms = int(segment["start_time"] * 1000)
            end_time_ms = int(segment["end_time"] * 1000)
            
            if not output_text or output_text == "...":
                logger(f"   ⚠️ Segment {i} has no translation text, skipping")
                continue
                
            logger(f"   📝 Text: '{output_text[:50]}...' ({start_time_ms/1000:.1f}s-{end_time_ms/1000:.1f}s)")

            selected_voice = next(
                (
                    item["selected_voice"]
                    for item in speaker_assignments
                    if item["speaker_label"] == segment["speaker_label"]
                ),
                FALLBACK_VOICE,
            )

            segment_details = {
                "character_type": segment["character_type"],
                "emotion": segment.get("emotion", "NEUTRAL"),
                "delivery_style": segment.get("delivery_style", "NORMAL"),
                "speaker_label": segment.get("speaker_label", "DEFAULT"),
                "pace": segment.get("pace", "NORMAL"),
                "clip_duration": end_time_ms - start_time_ms,
                "selected_voice": selected_voice,
                "output_path": os.path.join(output_dir, f"segment_{i}.wav"),
            }

            synthesized_path = segment_details['output_path']

            # Reuse existing audio segment if available and enabled
            if config.get("REUSE", False) and os.path.exists(synthesized_path):
                logger(f"   ✅ Reusing existing audio for segment {i}")
            else:
                # Save TTS prompt for first few segments as samples
                if i < 3 and file_manager:  # Save first 3 segments as samples
                    tts_prompt = self._build_tts_prompt(output_text, segment_details, config)
                    segment_info = {
                        'speaker_label': segment_details.get('speaker_label', 'unknown'),
                        'segment_number': i,
                        'text_preview': output_text[:50] + "..." if len(output_text) > 50 else output_text,
                        'character_type': segment_details.get('character_type'),
                        'emotion': segment_details.get('emotion'),
                        'delivery_style': segment_details.get('delivery_style'),
                        'timing': f"{start_time_ms/1000:.1f}s-{end_time_ms/1000:.1f}s"
                    }
                    
                    file_manager.save_prompt(
                        prompt_content=tts_prompt,
                        prompt_type="tts",
                        segment_info=segment_info,
                        logger=logger
                    )
                
                # Rate limiting before synthesis
                time.sleep(2)
                synthesized_path = self.synthesize_speech_with_gemini(
                    output_text, segment_details, config, logger
                )

            if synthesized_path and os.path.exists(synthesized_path):
                dub_segment = self._process_synthesized_audio(
                    synthesized_path, segment_details, output_dir, i, logger
                )
                if dub_segment:
                    # Check if segment timing is within background music duration
                    bg_duration_ms = len(background_music)
                    if start_time_ms >= bg_duration_ms:
                        logger(f"   ⚠️ Segment {i} start time ({start_time_ms/1000:.1f}s) beyond background duration ({bg_duration_ms/1000:.1f}s)")
                        logger(f"   🔧 Extending background music to accommodate segment")
                        # Extend background with silence if needed
                        extension_needed = start_time_ms + len(dub_segment) - bg_duration_ms
                        if extension_needed > 0:
                            final_vocal_track = final_vocal_track + AudioSegment.silent(duration=extension_needed)
                    
                    final_vocal_track = final_vocal_track.overlay(
                        dub_segment, position=start_time_ms
                    )
                    successful_syntheses += 1
                    logger(f"   ✅ Segment {i} overlaid at {start_time_ms/1000:.1f}s")
                else:
                    logger(f"   ❌ Segment {i} audio processing failed")
                
                # Don't remove reused files
                if not (config.get("REUSE", False) and os.path.exists(segment_details['output_path'])):
                     if synthesized_path != segment_details['output_path']: # Stretched audio
                        os.remove(synthesized_path)

            else:
                logger(f"⚠️ WARNING: Line {i} could not be synthesized.")
                if config.get("STRICT", False):
                    logger("❌ STRICT mode: Aborting due to TTS failure.")
                    return None

            print(f"   Line {i+1}/{total_lines} complete")

        if successful_syntheses == 0 and total_lines > 0:
            logger("❌ CRITICAL: No audio segments were successfully synthesized. Aborting.")
            return None

        # Log final summary
        logger(f"✅ Audio processing complete: {successful_syntheses}/{total_lines} segments successfully synthesized")
        logger(f"   📊 Final vocal track duration: {len(final_vocal_track)/1000:.1f}s")
        
        return final_vocal_track

    def _authenticate_gemini(self, config, logger):
        """Authenticate with Gemini API for TTS."""
        try:
            if config.get("USE_VERTEX_AI"):
                return genai.Client(
                    project=config["PROJECT_ID"], location=config["LOCATION"]
                )
            else:
                return genai.Client(api_key=config["GEMINI_API_KEY"])
        except Exception as e:
            logger(f"❌ TTS Authentication failed: {e}")
            return None

    def _build_tts_prompt(self, text, segment_details, config):
        """Build enhanced TTS prompt with naturalistic performance guidance."""
        
        # Character voice foundation
        voice_foundation = self._build_character_voice_foundation(segment_details, config)
        
        # Prosodic delivery instructions
        prosodic_guidance = self._build_prosodic_instructions(segment_details)
        
        # Contextual performance notes
        contextual_delivery = self._build_contextual_delivery(segment_details)
        
        # Naturalness and timing instructions
        naturalness_guidance = self._build_naturalness_instructions(segment_details)
        
        # Add accent enforcement for consistency
        accent_enforcement = self._build_accent_enforcement(config)
        
        # Add dubbing script compliance instructions
        script_compliance = self._build_script_compliance_instructions(segment_details)
        
        return f"""
        You are voicing {segment_details.get('speaker_label', 'the character')} in a professional film dubbing session.
        
        CHARACTER VOICE: {voice_foundation}
        
        ACCENT CONSISTENCY: {accent_enforcement}
        
        PROSODIC DELIVERY: {prosodic_guidance}
        
        PERFORMANCE CONTEXT: {contextual_delivery}
        
        SCRIPT COMPLIANCE: {script_compliance}
        
        NATURALNESS: {naturalness_guidance}
        
        TIMING: Deliver naturally within {segment_details['clip_duration']}ms, maintaining organic rhythm and flow.
        
        TEXT: "{text}"
        
        CRITICAL: Maintain consistent accent throughout. Perform this line as the character in this exact emotional moment, not as script reading.
        Focus on authentic human speech patterns, emotional authenticity, and strict adherence to accent specifications.
        """

    def _build_character_voice_foundation(self, segment_details, config):
        """Build character voice description with accent specification."""
        char_type = segment_details.get("character_type", "ADULT")
        language = config['OUTPUT_LANGUAGE']
        
        # Specify accent variant for English
        accent_specification = self._get_accent_specification(language)
        
        if char_type == "MALE":
            return f"Adult male {accent_specification} speaker with natural masculine vocal characteristics"
        elif char_type == "FEMALE":
            return f"Adult female {accent_specification} speaker with natural feminine vocal characteristics"
        elif char_type == "CHILD":
            return f"Young {accent_specification} speaker with age-appropriate higher pitch and youthful speech patterns"
        elif char_type == "ELDERLY":
            return f"Elderly {accent_specification} speaker with mature vocal characteristics and life experience"
        else:
            return f"Natural {accent_specification} speaker with conversational vocal characteristics"

    def _get_accent_specification(self, language):
        """Get accent specification for the language with Indian English default."""
        language_lower = language.lower()
        
        # Explicit accent mapping for English variants
        if language_lower in ['english', 'english (indian)', 'indian english']:
            return "Indian English"
        elif language_lower in ['english (british)', 'british english']:
            return "British English"
        elif language_lower in ['english (american)', 'american english']:
            return "American English"
        elif language_lower in ['english (australian)', 'australian english']:
            return "Australian English"
        elif language_lower in ['english (canadian)', 'canadian english']:
            return "Canadian English"
        
        # Default any unspecified English to Indian English
        elif 'english' in language_lower:
            return "Indian English"
        
        # For non-English languages, return as-is
        else:
            return language

    def _build_prosodic_instructions(self, segment_details):
        """Build detailed prosodic delivery guidance."""
        intonation = segment_details.get("intonation_pattern", "FALLING")
        voice_quality = segment_details.get("voice_quality", "MODAL")
        pace = segment_details.get("pace", "NORMAL")
        
        prosodic_notes = []
        
        # Intonation guidance
        if intonation == "RISING":
            prosodic_notes.append("Use rising intonation (↗) - questioning, uncertain, or list-like delivery")
        elif intonation == "FALLING":
            prosodic_notes.append("Use falling intonation (↘) - declarative, completion, authority")
        elif intonation == "RISE_FALL":
            prosodic_notes.append("Use rise-fall pattern (↗↘) - emphasis, contrast, significance")
        elif intonation == "FLAT":
            prosodic_notes.append("Use flat intonation (→) - monotone, boredom, or controlled emotion")
        
        # Voice quality guidance
        if voice_quality == "BREATHY":
            prosodic_notes.append("Breathy voice quality - intimate, tired, or sensual delivery")
        elif voice_quality == "CREAKY":
            prosodic_notes.append("Creaky voice quality - authority, low pitch, vocal fry characteristics")
        elif voice_quality == "TENSE":
            prosodic_notes.append("Tense voice quality - stress, anger, physical effort")
        else:
            prosodic_notes.append("Modal voice quality - natural, relaxed vocal delivery")
        
        # Pace guidance
        pace_guidance = {
            "VERY_SLOW": "Very deliberate, drawn-out delivery with dramatic emphasis",
            "SLOW": "Measured, thoughtful speech with careful articulation", 
            "NORMAL": "Standard conversational rhythm and timing",
            "FAST": "Quick, energetic delivery with increased tempo",
            "VERY_FAST": "Rapid, rushed delivery suggesting excitement or urgency",
            "IRREGULAR": "Varied pace within the segment for natural speech patterns"
        }
        prosodic_notes.append(pace_guidance.get(pace, "Natural conversational pace"))
        
        return " • ".join(prosodic_notes)

    def _build_contextual_delivery(self, segment_details):
        """Build contextual performance guidance."""
        emotion = segment_details.get("emotion", "NEUTRAL")
        emotion_intensity = segment_details.get("emotion_intensity", "MODERATE")
        delivery_style = segment_details.get("delivery_style", "NORMAL")
        relationship_context = segment_details.get("relationship_context", "EQUAL")
        scene_energy = segment_details.get("scene_energy", "CASUAL")
        
        context_notes = []
        
        # Emotional delivery
        if emotion != "NEUTRAL":
            intensity_modifier = {
                "MILD": "subtle",
                "MODERATE": "clear",
                "INTENSE": "strong"
            }.get(emotion_intensity, "moderate")
            context_notes.append(f"{intensity_modifier} {emotion.lower()} emotional coloring")
        
        # Delivery style specifics
        style_guidance = {
            "SHOUTING": "Loud, projected delivery with increased volume",
            "WHISPERING": "Quiet, intimate delivery with reduced volume",
            "CRYING": "Voice breaking with emotional distress",
            "PLEADING": "Urgent, desperate tone with begging quality",
            "LAUGHING": "Joyful delivery with laughter elements",
            "STORYTELLING": "Engaging, narrative rhythm",
            "EXPLAINING": "Clear, methodical delivery",
            "ARGUING": "Confrontational with sharp edges",
            "COMMANDING": "Authoritative, direct delivery"
        }
        if delivery_style in style_guidance:
            context_notes.append(style_guidance[delivery_style])
        
        # Relationship dynamics
        relationship_guidance = {
            "DOMINANT": "Authoritative, confident delivery from position of power",
            "SUBMISSIVE": "Deferential, respectful tone",
            "INTIMATE": "Close, personal delivery for private moment",
            "EQUAL": "Peer-level conversational delivery"
        }
        context_notes.append(relationship_guidance.get(relationship_context, "Natural conversational delivery"))
        
        # Scene energy
        energy_guidance = {
            "BUILDUP": "Increasing tension and anticipation in delivery",
            "CLIMAX": "Peak emotional intensity and drama",
            "RESOLUTION": "Releasing tension with settling energy",
            "CASUAL": "Low-stakes, relaxed conversational energy"
        }
        context_notes.append(energy_guidance.get(scene_energy, "Appropriate scene energy"))
        
        return " • ".join(context_notes)

    def _build_naturalness_instructions(self, segment_details):
        """Build naturalness and authentic speech guidance."""
        natural_pauses = segment_details.get("natural_pauses", [])
        prosodic_notes = segment_details.get("prosodic_notes", "")
        
        naturalness_guidance = []
        
        # Pause integration
        if natural_pauses:
            pause_types = ", ".join(natural_pauses)
            naturalness_guidance.append(f"Include natural {pause_types.lower().replace('_', ' ')} for authentic speech flow")
        
        # Prosodic notes from analysis
        if prosodic_notes:
            naturalness_guidance.append(f"Performance notes: {prosodic_notes}")
        
        # General naturalness
        naturalness_guidance.extend([
            "Maintain natural speech rhythm with organic timing",
            "Use authentic vocal expressions and micro-variations",
            "Avoid robotic or overly perfect pronunciation", 
            "Include natural vocal characteristics like slight hesitations or emphasis variations"
        ])
        
        return " • ".join(naturalness_guidance)

    def _build_accent_enforcement(self, config):
        """Build accent enforcement instructions."""
        language = config['OUTPUT_LANGUAGE']
        accent_specification = self._get_accent_specification(language)
        
        # Strong accent enforcement for English variants
        if 'english' in language.lower():
            return f"""
            MANDATORY: Use ONLY {accent_specification} pronunciation, intonation, and speech patterns.
            - Phonetic characteristics: Follow {accent_specification} vowel systems and consonant patterns
            - Rhythm and stress: Use {accent_specification} sentence stress and timing patterns
            - Intonation: Apply {accent_specification} rise-fall patterns and question intonation
            - Vocabulary: Use {accent_specification} word choices and expressions where appropriate
            - Consistency: Maintain the same accent throughout ALL segments for this character
            - NEVER mix accents: Do not slip into American, British, or other English variants
            """
        else:
            return f"Maintain consistent {accent_specification} pronunciation and speech characteristics throughout."

    def _build_script_compliance_instructions(self, segment_details):
        """Build instructions to ensure TTS follows dubbing script directions."""
        compliance_instructions = []
        
        # Emotion compliance
        emotion = segment_details.get("emotion", "NEUTRAL")
        emotion_intensity = segment_details.get("emotion_intensity", "MODERATE")
        if emotion != "NEUTRAL":
            compliance_instructions.append(f"EXPRESS {emotion.upper()} emotion with {emotion_intensity.lower()} intensity as specified")
        
        # Delivery style compliance  
        delivery_style = segment_details.get("delivery_style", "NORMAL")
        if delivery_style != "NORMAL":
            compliance_instructions.append(f"Use {delivery_style.upper()} delivery style as directed")
        
        # Prosodic compliance
        intonation = segment_details.get("intonation_pattern")
        if intonation:
            compliance_instructions.append(f"Apply {intonation.upper()} intonation pattern as analyzed")
        
        voice_quality = segment_details.get("voice_quality")
        if voice_quality and voice_quality != "MODAL":
            compliance_instructions.append(f"Use {voice_quality.upper()} voice quality as specified")
        
        pace = segment_details.get("pace", "NORMAL")
        if pace != "NORMAL":
            compliance_instructions.append(f"Maintain {pace.upper()} pace as directed")
        
        # Performance context compliance
        relationship_context = segment_details.get("relationship_context")
        if relationship_context:
            compliance_instructions.append(f"Reflect {relationship_context.upper()} relationship dynamic")
        
        scene_energy = segment_details.get("scene_energy")
        if scene_energy:
            compliance_instructions.append(f"Match {scene_energy.upper()} scene energy level")
        
        # Natural pauses compliance
        natural_pauses = segment_details.get("natural_pauses", [])
        if natural_pauses:
            pause_types = ", ".join([p.replace("_", " ").lower() for p in natural_pauses])
            compliance_instructions.append(f"Include identified {pause_types} as analyzed")
        
        # Prosodic notes compliance
        prosodic_notes = segment_details.get("prosodic_notes")
        if prosodic_notes:
            compliance_instructions.append(f"Follow specific performance notes: {prosodic_notes}")
        
        if compliance_instructions:
            return "FOLLOW SCRIPT ANALYSIS: " + " • ".join(compliance_instructions)
        else:
            return "Follow all emotional and delivery specifications from the dubbing script analysis."

    def _extract_audio_from_response(self, response, output_path, logger):
        """Extract audio data from Gemini API response with enhanced error details."""
        try:
            # Check response structure with detailed logging
            if not response:
                logger("❌ TTS response is None")
                return None
                
            if not hasattr(response, 'candidates') or not response.candidates:
                logger("❌ TTS response has no candidates")
                if hasattr(response, 'prompt_feedback'):
                    logger(f"   📋 Prompt feedback: {response.prompt_feedback}")
                return None
                
            if len(response.candidates) == 0:
                logger("❌ TTS response candidates list is empty")
                return None
                
            candidate = response.candidates[0]
            if not hasattr(candidate, 'content') or not candidate.content:
                logger("❌ TTS response candidate has no content")
                if hasattr(candidate, 'finish_reason'):
                    logger(f"   🛑 Finish reason: {candidate.finish_reason}")
                if hasattr(candidate, 'safety_ratings'):
                    logger(f"   🛡️ Safety ratings: {candidate.safety_ratings}")
                return None
                
            if not hasattr(candidate.content, 'parts') or not candidate.content.parts:
                logger("❌ TTS response content has no parts")
                return None
                
            if len(candidate.content.parts) == 0:
                logger("❌ TTS response content parts list is empty")
                return None
                
            part = candidate.content.parts[0]
            if not hasattr(part, 'inline_data') or not part.inline_data:
                logger("❌ TTS response part has no inline_data")
                if hasattr(part, 'text'):
                    logger(f"   📝 Part contains text instead of audio: {part.text[:100]}...")
                return None
                
            if not hasattr(part.inline_data, 'data') or not part.inline_data.data:
                logger("❌ TTS response inline_data has no audio data")
                if hasattr(part.inline_data, 'mime_type'):
                    logger(f"   🎵 MIME type: {part.inline_data.mime_type}")
                return None
            
            audio_data = part.inline_data.data
            
            # Validate audio data
            if len(audio_data) == 0:
                logger("❌ TTS audio data is empty (0 bytes)")
                return None
            
            # Log successful extraction
            logger(f"   ✅ Extracted {len(audio_data)} bytes of audio data")
            
            # Write audio file
            self._write_wave_file(output_path, audio_data)
            
            # Verify file was written successfully
            if not os.path.exists(output_path):
                logger(f"❌ Failed to write audio file: {output_path}")
                return None
                
            file_size = os.path.getsize(output_path)
            if file_size == 0:
                logger(f"❌ Written audio file is empty: {output_path}")
                return None
                
            logger(f"   📁 Audio file written: {file_size} bytes")
            return output_path
            
        except Exception as e:
            logger(f"❌ Error extracting audio from TTS response: {e}")
            logger(f"   🔍 Response type: {type(response)}")
            if hasattr(response, '__dict__'):
                logger(f"   📋 Response attributes: {list(response.__dict__.keys())}")
            return None

    def _write_wave_file(self, filename, pcm, channels=1, rate=24000, sample_width=2):
        """Write PCM data to a wave file."""
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(pcm)

    def _process_synthesized_audio(
        self, synthesized_path, segment_details, output_dir, segment_index, logger
    ):
        """Process synthesized audio including time stretching if needed."""
        try:
            with open(synthesized_path, "rb") as f:
                dub_segment = AudioSegment.from_wav(f)

                # Apply time stretching if needed
                original_duration_ms = len(dub_segment)
                target_duration_ms = segment_details["clip_duration"]

                if target_duration_ms > 0 and original_duration_ms > 0:
                    speed_ratio = original_duration_ms / target_duration_ms
                    if speed_ratio > 1.5:
                        speed_ratio = 1.3

                    if abs(1 - speed_ratio) > 0.05:
                        stretched_audio_path = os.path.join(
                            output_dir, f"segment_{segment_index}_stretched.wav"
                        )

                        y, sr = sf.read(synthesized_path)
                        y_stretched = rb.time_stretch(y, sr, speed_ratio)
                        sf.write(stretched_audio_path, y_stretched, sr)

                        dub_segment = AudioSegment.from_wav(stretched_audio_path)
                        os.remove(stretched_audio_path)

                return dub_segment

        except Exception as e:
            logger(
                f"   ⚠️ Could not time-stretch line {segment_index}, using original audio."
            )
            # Return original segment without time stretching
            with open(synthesized_path, "rb") as f:
                return AudioSegment.from_wav(f)
