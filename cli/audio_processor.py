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
        """Synthesize speech using Gemini TTS API."""
        logger(
            f"   Synthesizing: '{text[:40]}...' for {segment_details['speaker_label']}"
        )

        try:
            # Authenticate with Gemini API
            client = self._authenticate_gemini(config, logger)
            if not client:
                return None

            # Build voice description and prompt
            full_prompt = self._build_tts_prompt(text, segment_details, config)

            # Generate speech
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
            return self._extract_audio_from_response(
                response, segment_details["output_path"], logger
            )

        except Exception as e:
            logger(f"❌ Error synthesizing speech with Gemini: {e}")
            return None

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
    ):
        """Process all audio segments for dubbing."""
        final_vocal_track = AudioSegment.silent(duration=len(background_music))
        output_lang_key = f"{config['OUTPUT_LANGUAGE']}_translation"
        total_lines = len(dubbing_script)
        successful_syntheses = 0

        FALLBACK_VOICE = "Leda"

        for i, segment in enumerate(dubbing_script):
            output_text = segment.get(output_lang_key, "...")
            start_time_ms = int(segment["start_time"] * 1000)
            end_time_ms = int(segment["end_time"] * 1000)

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
                    final_vocal_track = final_vocal_track.overlay(
                        dub_segment, position=start_time_ms
                    )
                    successful_syntheses += 1
                
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
        """Build the prompt for TTS generation."""
        voice_description = f"a standard '{config['OUTPUT_LANGUAGE']}' voice"
        if segment_details["character_type"] == "MALE":
            voice_description = f"a standard '{config['OUTPUT_LANGUAGE']}' MALE voice"
        elif segment_details["character_type"] == "FEMALE":
            voice_description = f"a standard '{config['OUTPUT_LANGUAGE']}' FEMALE voice"
        elif segment_details["character_type"] == "CHILD":
            voice_description = (
                f"a higher-pitched '{config['OUTPUT_LANGUAGE']}' child's voice"
            )

        style_description = ""
        if segment_details["delivery_style"] == "SHOUTING":
            style_description = " in a loud, shouting tone"
        elif segment_details["delivery_style"] == "WHISPERING":
            style_description = " in a soft, whispering tone"
        elif segment_details["delivery_style"] == "NORMAL":
            style_description = "in a normal tone"
        elif segment_details["emotion"] not in ["NEUTRAL", "NORMAL"]:
            style_description = f" in a {segment_details['emotion'].lower()} tone"

        instruction = f"You are a movie dubbing expert and need to deliver audio as professional editor. Maintain the duration of the output audio within {segment_details['clip_duration']} milliseconds."

        return f"{instruction}. Using {voice_description} {style_description} with {segment_details['emotion']} emotion, {segment_details['speaker_label']} says the following at {segment_details['pace']} speed: {text}"

    def _extract_audio_from_response(self, response, output_path, logger):
        """Extract audio data from Gemini API response."""
        if (
            not response.candidates
            or not response.candidates[0].content
            or not response.candidates[0].content.parts
        ):
            logger(
                f"❌ Gemini TTS returned an empty or invalid response for the text. Skipping."
            )
            return None

        audio_data = response.candidates[0].content.parts[0].inline_data.data
        if not audio_data:
            logger(f"❌ Gemini did not return audio data for the text.")
            return None

        self._write_wave_file(output_path, audio_data)
        return output_path

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
