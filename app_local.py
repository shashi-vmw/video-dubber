import os
import streamlit as st
import google.genai as genai
import time
import json
from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips
from pydub import AudioSegment
import subprocess
from google.genai import types
import wave
import tempfile
from google.cloud import storage
import datetime
import sys

# --- UI HELPER DATA ---

# A curated list of common languages for the dropdowns
LANGUAGES = [
    "Arabic", "Bengali", "Chinese", "Dutch", "English", "French", "German",
    "Hindi", "Indonesian", "Italian", "Japanese", "Korean", "Malayalam",
    "Marathi", "Polish", "Portuguese", "Punjabi", "Russian", "Spanish",
    "Tamil", "Telugu", "Turkish", "Ukrainian", "Urdu", "Vietnamese"
]

# List of GCP regions for the dropdown
GCP_REGIONS = [
    "global", "africa-south1", "us-central1", "us-east1", "us-east4",
    "us-east5", "us-south1", "us-west1", "us-west2", "us-west3", "us-west4",
    "asia-east1", "asia-east2", "asia-northeast1", "asia-northeast2",
    "asia-northeast3", "asia-south1", "asia-south2", "asia-southeast1",
    "asia-southeast2", "australia-southeast1", "australia-southeast2",
    "europe-central2", "europe-north1", "europe-southwest1", "europe-west1",
    "europe-west2", "europe-west3", "europe-west4", "europe-west6",
    "europe-west8", "europe-west9", "europe-west10", "europe-west12",
    "me-central1", "me-central2", "me-west1", "southamerica-east1",
    "southamerica-west1"
]

# --- VOICE CONFIG (From original script) ---
MALE_VOICE_LIST = [
    'Achird', 'Algenib', 'Algieba', 'Alnilam', 'Charon', 'Enceladus',
    'Fenrir', 'Iapetus', 'Orus', 'Puck', 'Rasalgethi', 'Sadachbia',
    'Sadaltager', 'Schedar', 'Umbriel', 'Zubenelgenubi'
]
FEMALE_VOICE_LIST = [
    'Achernar', 'Aoede', 'Autonoe', 'Callirrhoe', 'Despina', 'Erinome',
    'Gacrux', 'Kore', 'Laomedeia', 'Leda', 'Pulcherrima', 'Sulafat',
    'Vindemiatrix', 'Zephyr'
]
CHILD_VOICE_LIST = ['Leda', 'Kore']
FALLBACK_VOICE = 'Leda'

# --- GCS HELPER FUNCTIONS ---
def get_gcs_client():
    return storage.Client()

def list_gcs_buckets(client):
    try:
        return [bucket.name for bucket in client.list_buckets()]
    except Exception as e:
        st.error(f"Failed to list buckets: {e}")
        return []

def list_gcs_files(client, bucket_name, prefix=""):
    try:
        blobs = client.list_blobs(bucket_name, prefix=prefix)
        # Filter for video files
        video_extensions = ('.mp4', '.mov', '.avi', '.mkv')
        return [blob.name for blob in blobs if blob.name.lower().endswith(video_extensions)]
    except Exception as e:
        st.error(f"Failed to list files in bucket {bucket_name}: {e}")
        return []

def upload_blob(client, bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        return f"gs://{bucket_name}/{destination_blob_name}"
    except Exception as e:
        st.error(f"Failed to upload file: {e}")
        return None

def download_blob(client, bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        return True
    except Exception as e:
        st.error(f"Failed to download file: {e}")
        return False

# --- STATUS LOGGER CLASS FOR UI UPDATES ---
class StatusLogger:
    """A helper class to log status updates to the Streamlit UI."""
    def __init__(self):
        self.status_area = st.empty()
        self.log_messages = []

    def log(self, message, type="info"):
        # Prepend emoji based on type for better visual cues
        if "✅" in message or "🎉" in message:
            type = "success"
        elif "❌" in message or "⚠️" in message:
            type = "warning"

        self.log_messages.append(message)
        
        # Display messages in styled containers
        with self.status_area.container():
            st.subheader("⚙️ Processing Status")
            for msg in reversed(self.log_messages): # Show newest first
                if "✅" in msg or "🎉" in msg:
                    st.success(msg)
                elif "❌" in msg or "⚠️" in msg:
                    st.warning(msg)
                else:
                    st.info(msg)

# --- CORE LOGIC (Refactored to accept config and logger) ---

def get_dubbing_script_from_video(video_uri, config, logger):
    """
    Analyzes a video from GCS URI to produce a dubbing script.
    """
    try:
        if config["USE_VERTEX_AI"]:
            logger.log(f"Authenticating with Vertex AI (Project: {config['PROJECT_ID']}, Location: {config['LOCATION']})")
            client = genai.Client(vertexai=True, project=config["PROJECT_ID"], location=config["LOCATION"])
        else:
            logger.log("Authenticating with Google API Key")
            client = genai.Client(api_key=config["GOOGLE_API_KEY"])
    except Exception as e:
        logger.log(f"❌ Authentication failed: {e}")
        return None

    # Get local video duration for the prompt
    with VideoFileClip(config['local_video_path']) as clip:
        video_duration = clip.duration
        logger.log(f"ℹ️ Media duration: {video_duration:.2f} seconds.")

    # Method A: Pass GCS URI directly in contents (Best for Vertex AI)
    logger.log(f"🤖 Sending request to {config['MODEL_NAME']} for analysis...")
    
    video_part = types.Part.from_uri(
        file_uri=video_uri,
        mime_type="video/mp4" # Assuming mp4, could be dynamic
    )

    base_prompt = f"""
    You are an expert voice director and video producer creating a script for dubbing.
    Analyze the provided video file's audio track with extreme detail. Your goal is to capture the complete performance, including the emotional and dramatic context.

    Follow these steps precisely:
    1.  **Speaker Diarization**: 
        *   Identify every distinct speaker and assign a consistent unique label (e.g., "SPEAKER_01", "SPEAKER_02").
        *   **CRITICAL**: Ensure the same character ALWAYS has the same label throughout the entire video. Do not switch labels mid-video.
    2.  **Character Classification**: Classify as MALE, FEMALE, or CHILD.
    3.  **Emotional Analysis**: HAPPY, SAD, ANGRY, SURPRISED, FEARFUL, NEUTRAL.
    4.  **Delivery Style**: NORMAL, SHOUTING, WHISPERING.
    5.  **Transcription & Translation**:
        *   Provide original '{config["INPUT_LANGUAGE"]}' and meaningful '{config["OUTPUT_LANGUAGE"]}' translation.
        *   **CRITICAL**: You MUST capture and include non-verbal vocalizations and interjections (e.g., sighs, gasps, laughs, grunts, "umm", "ah") in the translation to preserve authenticity.
    6.  **Pace**: NORMAL, FAST, SLOW, etc.
    7.  **PRECISE TIMING (Voice Onset)**:
        *   **CRITICAL**: The `start_time` must correspond to the **EXACT instant the character starts speaking** (lip movement/voice onset), NOT when the scene starts or when they appear on screen.
        *   Do NOT include leading silence, breaths, or reaction shots in the segment duration.
        *   If a character appears at 0:00 but speaks at 0:05, `start_time` MUST be `5.0`.
    8.  **Syllable Ratio & Timing Optimization**:
        *   Analyze the syllable density difference between {config["INPUT_LANGUAGE"]} and {config["OUTPUT_LANGUAGE"]}.
        *   The translation MUST naturally fit the duration of the original speech.
        *   Condense the translation (use concise synonyms) if the target language is verbose or the speaker is fast.
    9.  **NON-NEGOTIABLE TIMESTAMP CONVERSION**:
        *   **CRITICAL**: You MUST convert all minutes to total seconds.
        *   **FORMULA**: Total Seconds = (Minutes * 60) + Seconds.
        *   **MANDATORY EXAMPLES**:
            *   0:59 -> `59.0`
            *   1:00 -> `60.0`
            *   1:01 -> `61.0` (NOT 101)
            *   1:05 -> `65.0` (NOT 105)
            *   1:30 -> `90.0` (NOT 130)
            *   2:00 -> `120.0`
        *   **TOTAL LIMIT**: The media duration is exactly {video_duration:.2f} seconds. No timestamp can exceed this.
    10. **Output Format**: Valid JSON array of objects.
        {{
          "start_time": float,
          "end_time": float,
          "speaker_label": "string",
          "character_type": "string",
          "emotion": "string",
          "delivery_style": "string",
          "original_transcript": "string",
          "{config["OUTPUT_LANGUAGE"]}_translation": "string",
          "pace": "string"
        }}
    """
    
    max_retries = 3
    current_attempt = 0
    dubbing_script = None
    
    while current_attempt < max_retries:
        if current_attempt > 0:
            logger.log(f"🔄 Retry Attempt {current_attempt + 1}/{max_retries} due to validation error...")

        try:
            response = client.models.generate_content(
                model=config['MODEL_NAME'],
                contents=[video_part, "\n\n", base_prompt]
            )
            
            json_text = response.text.strip().lstrip("```json").rstrip("```")
            raw_script = json.loads(json_text)
            
            # --- Duration Validation Loop ---
            if not raw_script:
                raise ValueError("Empty script returned.")
                
            # Basic sort to find the true end
            raw_script.sort(key=lambda x: float(x.get('end_time', 0)))
            last_timestamp = float(raw_script[-1]['end_time'])
            
            if last_timestamp > video_duration + 2.0: # 2 second tolerance for float drift
                error_msg = f"Generated script duration ({last_timestamp:.2f}s) exceeds video duration ({video_duration:.2f}s)."
                logger.log(f"❌ {error_msg}")
                # Add specific error to prompt for next iteration
                base_prompt += f"\n\n🚨 **CRITICAL ERROR IN PREVIOUS ATTEMPT**: Your previous script ended at {last_timestamp}s, but the video is ONLY {video_duration}s long. You likely made a minute-to-second conversion error (e.g. 1:30 is 90s, not 130s). RE-CALCULATE TIMESTAMPS CAREFULLY."
                current_attempt += 1
                continue
            
            # If we get here, the script is valid duration-wise
            dubbing_script = raw_script
            break

        except Exception as e:
            logger.log(f"❌ Gemini Analysis failed (Attempt {current_attempt + 1}): {e}")
            current_attempt += 1
            time.sleep(2)

    if not dubbing_script:
        logger.log("❌ Failed to generate a valid script after multiple attempts.")
        return None

    try:
        # Log Raw Script for Debugging
        logger.log(f"📝 Raw Gemini Output: {json.dumps(dubbing_script, indent=2)}") 
        
        # --- Validate and Fix Script ---
        dubbing_script = validate_and_fix_script(dubbing_script, video_duration, logger)
        
        # --- NEW: Refine Script with Gemini Loop ---
        dubbing_script = validate_and_refine_script(dubbing_script, config, logger)
        
        # --- FINAL SAFETY CHECK: Re-validate duration after refinement ---
        dubbing_script = validate_and_fix_script(dubbing_script, video_duration, logger)
        
        logger.log("✅ Successfully received and parsed the dubbing script from Gemini.")
        with st.expander("Show Generated Dubbing Script (JSON)"):
            st.json(dubbing_script)
        return dubbing_script
    except (json.JSONDecodeError, IndexError, AttributeError) as e:
        logger.log(f"❌ Failed to parse JSON from Gemini's response: {e}")
        return None

def validate_and_fix_script(script, video_duration, logger):
    """
    Validates and corrects the dubbing script.
    Checks for chronology, overlaps, small gaps, and huge missing chunks.
    """
    if not script:
        return []

    # 1. Convert timestamps to float and Sort
    valid_script = []
    for seg in script:
        try:
            seg['start_time'] = float(seg['start_time'])
            seg['end_time'] = float(seg['end_time'])
            
            # Enforce minimum duration of 300ms
            if seg['end_time'] - seg['start_time'] < 0.3:
                 seg['end_time'] = seg['start_time'] + 0.3

            if seg['end_time'] > seg['start_time']:
                valid_script.append(seg)
        except (ValueError, TypeError):
            logger.log(f"⚠️ Skipping invalid segment timestamps: {seg}")
            continue
            
    valid_script.sort(key=lambda x: x['start_time'])

    # 2. Fix Overlaps and Gaps
    fixed_script = []
    for i, current_seg in enumerate(valid_script):
        if i == 0:
            fixed_script.append(current_seg)
            continue
        
        prev_seg = fixed_script[-1]
        
        # Check for overlap
        if current_seg['start_time'] < prev_seg['end_time']:
            overlap = prev_seg['end_time'] - current_seg['start_time']
            logger.log(f"⚠️ Overlap detected between segments ({overlap:.3f}s). Adjusting prev end time.")
            prev_seg['end_time'] = current_seg['start_time']
            
            if prev_seg['end_time'] <= prev_seg['start_time']:
                 fixed_script.pop()
        
        # Check for tiny gaps (< 100ms) - Merge
        elif 0 < (current_seg['start_time'] - prev_seg['end_time']) < 0.1:
             prev_seg['end_time'] = current_seg['start_time']
        
        # Check for HUGE gaps (> 2s) - Warn
        gap = current_seg['start_time'] - prev_seg['end_time']
        if gap > 2.0:
            logger.log(f"⚠️ Large silence detected: {gap:.2f}s gap between {prev_seg['end_time']:.2f}s and {current_seg['start_time']:.2f}s. Potentially missed dialogue?")

        fixed_script.append(current_seg)

    # 3. Check for Early Termination
    if fixed_script:
        last_end = fixed_script[-1]['end_time']
        if video_duration and (video_duration - last_end > 5.0):
             logger.log(f"⚠️ Script ends early! Last segment at {last_end:.2f}s, but video is {video_duration:.2f}s long. Missed {video_duration - last_end:.2f}s at the end.")

    return fixed_script

def validate_and_refine_script(script, config, logger):
    """
    Invokes Gemini again to validate the generated dubbing script for accuracy and timing.
    """
    logger.log("🕵️ Validating and refining script with Gemini...")
    
    try:
        if config["USE_VERTEX_AI"]:
            client = genai.Client(vertexai=True, project=config["PROJECT_ID"], location=config["LOCATION"])
        else:
            client = genai.Client(api_key=config["GOOGLE_API_KEY"])
            
        prompt = f"""
        You are a Quality Assurance expert for movie dubbing. 
        Review the following JSON dubbing script (Input Language: {config['INPUT_LANGUAGE']}, Output Language: {config['OUTPUT_LANGUAGE']}).
        
        Your Goal: Ensure the translation is accurate, natural, and strictly fits the time constraints.
        
        Instructions:
        1. Check if the '{config['OUTPUT_LANGUAGE']}_translation' conveys the meaning of 'original_transcript' accurately.
        2. Check if the translation is too long for the duration (end_time - start_time). If so, shorten it using synonyms or rephrasing.
        3. Ensure there are no hallucinated segments.
        4. **CRITICAL**: Do NOT change the `speaker_label` of any segment. You must preserve the exact speaker identifiers from the input script to ensure voice consistency.
        5. Return the CORRECTED script in the exact same JSON format. If a segment is perfect, return it as is.
        
        Input Script:
        {json.dumps(script)}
        
        Output:
        Provide ONLY the valid JSON array.
        """
        
        response = client.models.generate_content(
            model=config['MODEL_NAME'],
            contents=[prompt]
        )
        
        json_text = response.text.strip().lstrip("```json").rstrip("```")
        refined_script = json.loads(json_text)
        
        logger.log("✅ Script validation complete.")
        return refined_script

    except Exception as e:
        logger.log(f"⚠️ Script validation failed (using original script): {e}")
        return script


def extract_audio(video_path, audio_path, logger):
    logger.log(f"🎥 Extracting audio from '{os.path.basename(video_path)}'...")
    try:
        with VideoFileClip(video_path) as video_clip:
            video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le', logger=None)
        logger.log(f"✅ Audio extracted to '{os.path.basename(audio_path)}'")
        return audio_path
    except Exception as e:
        logger.log(f"❌ Error extracting audio: {e}")
        return None

def separate_background_music(audio_path, output_dir, logger):
    logger.log("🎶 Separating background music with Demucs... (This might take a while)")
    try:
        command = [
            sys.executable, "-m", "demucs.separate", "-n", "htdemucs",
            "-o", str(output_dir), "--two-stems", "vocals", str(audio_path)
        ]
        # Use st.spinner for long-running processes
        with st.spinner('Demucs is separating audio tracks... Please wait.'):
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        
        audio_filename = os.path.splitext(os.path.basename(audio_path))[0]
        model_name = "htdemucs"
        background_path = os.path.join(output_dir, model_name, audio_filename, "no_vocals.wav")

        if os.path.exists(background_path):
            logger.log("✅ Background music separated successfully.")
            return background_path
        else:
            logger.log("❌ Demucs did not produce the background music file.")
            if result.stderr:
                logger.log(f"Demucs Error: {result.stderr}")
            return None
    except Exception as e:
        logger.log(f"❌ An error occurred during audio separation: {e}")
        return None

def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)

def synthesize_speech_with_gemini(text, segment_details, config, logger):
    logger.log(f"   Synthesizing: '{text[:40]}...' for {segment_details['speaker_label']}")
    
    try:
        if config["USE_VERTEX_AI"]:
            client = genai.Client(vertexai=True, project=config["PROJECT_ID"], location=config["LOCATION"])
        else:
            client = genai.Client(api_key=config["GOOGLE_API_KEY"])
    except Exception as e:
        logger.log(f"❌ TTS Authentication failed: {e}")
        return None

    # --- Enhanced Prompting based on Gemini API Guide ---
    # 1. Audio Profile
    audio_profile = f"Speaker: {segment_details['speaker_label']}, Gender: {segment_details['character_type']}, Language: {config['OUTPUT_LANGUAGE']}."
    
    # 2. Scene
    scene_description = f"Movie dubbing session. The character is experiencing {segment_details['emotion']}."

    # 3. Director's Notes
    style_instruction = f"Emotion: {segment_details['emotion']}."
    if segment_details['delivery_style'] != "NORMAL":
        style_instruction += f" Delivery: {segment_details['delivery_style']} (act this out)."
    
    pace_instruction = f"Pace: {segment_details['pace']}."
    
    technical_instruction = f"Timing: The output audio duration should ideally fit within {segment_details['clip_duration']} milliseconds."

    full_prompt = f"""
    Audio Profile: {audio_profile}
    Scene: {scene_description}
    Director's Notes: {style_instruction} {pace_instruction} {technical_instruction}
    
    Text: {text}
    """

    try:
        response = client.models.generate_content(
        model=config['TTS_MODEL'],
        contents=full_prompt,
        config=types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
         voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
               voice_name=segment_details['selected_voice'],
               )
            )
        ),
        ))

        if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
            logger.log(f"❌ Gemini returned an empty or invalid response for the text. Skipping.")
            return None

        audio_data = response.candidates[0].content.parts[0].inline_data.data
        if not audio_data:
            logger.log(f"❌ Gemini did not return audio data for the text.")
            return None

        wave_file(segment_details['output_path'], audio_data)
        return segment_details['output_path']
    except Exception as e:
        logger.log(f"❌ Error synthesizing speech with Gemini: {e}")
        return None

def merge_audio_with_video(video_path, audio_path, output_path, logger):
    logger.log(f"🎬 Merging final audio with video...")
    try:
        with VideoFileClip(video_path) as video_clip, AudioFileClip(audio_path) as audio_clip:
            video_clip = VideoFileClip(video_path)
            audio_clip = AudioFileClip(audio_path)
            video_clip.audio = audio_clip
            final_clip = video_clip
            final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
        logger.log(f"🎉 Final video saved successfully to '{os.path.basename(output_path)}'")
        return output_path
    except Exception as e:
        logger.log(f"❌ An error occurred during video merging: {e}")
        return None

def assign_specific_voices(dubbing_script):
    """
    Extracts unique speakers from the script and assigns an initial voice.
    This list will be presented to the user for confirmation/editing.
    """
    speaker_info = {}
    for item in dubbing_script:
        speaker_label = item['speaker_label']
        if speaker_label not in speaker_info:
            speaker_info[speaker_label] = item['character_type']

    voice_indices = {'MALE': 0, 'FEMALE': 0, 'CHILD': 0}
    speaker_voice_assignments = []
    
    # Sort for consistency
    sorted_speakers = sorted(speaker_info.keys())

    for speaker in sorted_speakers:
        char_type = speaker_info[speaker]
        selected_voice = FALLBACK_VOICE
        
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
            
        speaker_voice_assignments.append({
            "speaker_label": speaker,
            "character_type": char_type,
            "selected_voice": selected_voice
        })
        
    return speaker_voice_assignments

def speed_up_audio(input_path, output_path, speed_factor):
    """Speeds up audio using ffmpeg's atempo filter."""
    try:
        # ffmpeg's atempo filter supports 0.5 to 2.0. If we need more, we might need chaining, 
        # but 1.5x is our cap, so one pass is enough.
        command = [
            "ffmpeg", "-y", "-i", input_path,
            "-filter:a", f"atempo={speed_factor}",
            "-vn", output_path
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False
    except Exception as e:
        print(f"Error changing speed: {e}")
        return False

def step1_generate_script(local_video_path, gcs_video_uri, config, logger):
    """
    Step 1: Audio Extraction, Background Separation, Script Generation.
    Returns (dubbing_script, background_track_path, original_audio_path)
    """
    base_name = os.path.splitext(os.path.basename(local_video_path))[0]
    
    # We use a persistent output directory in session state or fixed path
    output_dir = os.path.join("/Users/shashirr/Shashi/scvideos", f"output_{base_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Extract Audio from Local File
    original_audio_path = os.path.join(output_dir, f"{base_name}_original_audio.wav")
    if not os.path.exists(original_audio_path):
        extract_audio(local_video_path, original_audio_path, logger)
    else:
        logger.log("ℹ️ Using existing extracted audio.")
    
    # 2. Separate Background Music (Locally)
    background_track_path = os.path.join(output_dir, "separated", "htdemucs", f"{base_name}_original_audio", "no_vocals.wav")
    if not os.path.exists(background_track_path):
        background_track_path = separate_background_music(original_audio_path, os.path.join(output_dir, "separated"), logger)
    else:
        logger.log("ℹ️ Using existing background track.")
    
    # 3. Analyze Video with Gemini (Using GCS URI)
    config['local_video_path'] = local_video_path # Ensure local path is in config for duration check
    dubbing_script = get_dubbing_script_from_video(gcs_video_uri, config, logger)
    
    if not dubbing_script:
        logger.log("❌ Aborting due to failure in script generation.")
        return None, None, None

    return dubbing_script, background_track_path, output_dir


def step2_synthesize_video(local_video_path, dubbing_script, speaker_assignments, background_track_path, output_dir, config, logger):
    """
    Step 2: TTS Synthesis, Sync, and Merging.
    """
    base_name = os.path.splitext(os.path.basename(local_video_path))[0]
    
    logger.log(f"🔊 Initializing {config['TTS_MODEL']} for Text-to-Speech...")
    logger.log("🎤 Starting speech synthesis with Gemini...")

    with VideoFileClip(local_video_path) as clip:
        video_duration_ms = int(clip.duration * 1000)

    if background_track_path and os.path.exists(background_track_path):
        background_music = AudioSegment.from_wav(background_track_path)
        if len(background_music) < video_duration_ms:
             background_music += AudioSegment.silent(duration=video_duration_ms - len(background_music)) # Extend silence if shorter
    else:
        logger.log("⚠️ Could not separate background music. Using a silent background.")
        background_music = AudioSegment.silent(duration=video_duration_ms)

    final_vocal_track = AudioSegment.silent(duration=len(background_music))
    
    # Use the USER-CONFIRMED speaker assignments
    logger.log("Using confirmed voice assignments.")

    output_lang_key = f"{config['OUTPUT_LANGUAGE']}_translation"
    total_segments = len(dubbing_script)
    progress_bar = st.progress(0, text="Synthesizing audio segments...")
    
    # --- Timeline Reconstruction Lists ---
    final_video_clips = []
    final_audio_segments = [] # List of AudioSegments to be concatenated
    
    # Load original video for clipping
    original_video = VideoFileClip(local_video_path)
    last_pos = 0.0
    
    # Background music logic (ensure we have a valid AudioSegment)
    if background_track_path and os.path.exists(background_track_path):
         bg_audio_full = AudioSegment.from_wav(background_track_path)
    else:
         bg_audio_full = AudioSegment.silent(duration=int(original_video.duration * 1000))

    for i, segment in enumerate(dubbing_script):
        output_text = segment.get(output_lang_key, "...")
        start_time = segment['start_time']
        end_time = segment['end_time']
        start_time_ms = int(start_time * 1000)
        end_time_ms = int(end_time * 1000)
        
        # --- 1. Handle GAP (Video + Background Audio) ---
        if start_time > last_pos:
            gap_duration = start_time - last_pos
            if gap_duration > 0:
                # Video Gap
                video_gap = original_video.subclipped(last_pos, start_time)
                final_video_clips.append(video_gap)
                
                # Audio Gap (Background only)
                gap_start_ms = int(last_pos * 1000)
                gap_end_ms = int(start_time * 1000)
                audio_gap = bg_audio_full[gap_start_ms:gap_end_ms]
                final_audio_segments.append(audio_gap)

        # --- 2. Handle SPEECH SEGMENT ---
        selected_voice = next((item['selected_voice'] for item in speaker_assignments if item['speaker_label'] == segment['speaker_label']), FALLBACK_VOICE)
        
        # Target duration for synthesis is ideally the original duration
        orig_duration_ms = end_time_ms - start_time_ms
        
        segment_details = {
            'character_type': segment['character_type'],
            'emotion': segment.get('emotion', 'NEUTRAL'),
            'delivery_style': segment.get('delivery_style', 'NORMAL'),
            'speaker_label': segment.get('speaker_label', 'DEFAULT'),
            'pace': segment.get('pace', 'NORMAL'),
            'clip_duration': orig_duration_ms, 
            'selected_voice': selected_voice,
            'output_path': os.path.join(output_dir, f"segment_{i}.wav")
        }
        
        time.sleep(1.5) # Rate limit
        synthesized_path = synthesize_speech_with_gemini(output_text, segment_details, config, logger)
        
        if synthesized_path and os.path.exists(synthesized_path):
            with open(synthesized_path, "rb") as f:
                dub_segment = AudioSegment.from_wav(f)
            
            dub_duration_ms = len(dub_segment)
            orig_duration_sec = orig_duration_ms / 1000.0
            dub_duration_sec = dub_duration_ms / 1000.0
            
            # --- ADAPTIVE SYNC (Principle 3) ---
            ratio = dub_duration_sec / orig_duration_sec if orig_duration_sec > 0 else 1.0
            
            # Default: Use original video clip
            video_clip = original_video.subclipped(start_time, end_time)
            
            # Background audio for this segment (sliced from original)
            bg_segment = bg_audio_full[start_time_ms:end_time_ms]

            if 0.9 <= ratio <= 1.1:
                # Case A: Close match. Just stretch/squeeze audio slightly or leave as is.
                # Current pref: speed up audio if too long, leave if short (silence padding)
                if ratio > 1.0:
                     # Speed up audio to match video exactly
                     speed_factor = ratio
                     logger.log(f"🔹 Segment {i}: Slight audio speedup ({speed_factor:.2f}x)")
                     
                     fast_path = os.path.join(output_dir, f"segment_{i}_fast.wav")
                     if speed_up_audio(synthesized_path, fast_path, speed_factor):
                         with open(fast_path, "rb") as f_fast:
                             dub_segment = AudioSegment.from_wav(f_fast)
                         os.remove(fast_path)
                
                # If audio is shorter, we just use it (it will have silence at end if we use overlay on a longer bg)
                # Mix dub with background
                mixed_segment = bg_segment.overlay(dub_segment)
                final_audio_segments.append(mixed_segment)
                final_video_clips.append(video_clip)

            elif ratio > 1.1:
                # Case B: Audio is SIGNIFICANTLY longer.
                # Action: Slow down VIDEO to match Audio.
                logger.log(f"🐢 Segment {i}: slowing down VIDEO to match audio (Ratio {ratio:.2f}x).")
                
                # 1. Slow down video
                # speedx factor: < 1 means slow down. We want new duration = dub_duration.
                # factor = orig / dub = 1 / ratio
                video_factor = 1.0 / ratio
                video_clip = video_clip.with_speed_scaled(video_factor)
                final_video_clips.append(video_clip)
                
                # 2. Handle Audio
                # We have the long dub segment. We need background of same length.
                # We can't easily time-stretch background without artifacts using pure pydub.
                # Strategy: Loop the background segment to cover the duration? Or just fade out?
                # Let's loop with crossfade to extend it.
                extended_bg = bg_segment * int(ratio + 2) # Repeat enough times
                extended_bg = extended_bg[:dub_duration_ms] # Trim to exact length
                
                mixed_segment = extended_bg.overlay(dub_segment)
                final_audio_segments.append(mixed_segment)

            else: # ratio < 0.9
                # Case C: Audio is SIGNIFICANTLY shorter.
                # Action: Speed up VIDEO to match Audio.
                logger.log(f"🐇 Segment {i}: speeding up VIDEO to match audio (Ratio {ratio:.2f}x).")
                
                video_factor = 1.0 / ratio
                video_clip = video_clip.with_speed_scaled(video_factor)
                final_video_clips.append(video_clip)
                
                # Audio: Shorten background to match dub
                bg_segment = bg_segment[:dub_duration_ms]
                mixed_segment = bg_segment.overlay(dub_segment)
                final_audio_segments.append(mixed_segment)

            os.remove(synthesized_path)
        
        else:
            # Fallback if synthesis failed: Use original video + silent audio (or just bg)
            logger.log(f"⚠️ Segment {i} failed. Using original video + background only.")
            final_video_clips.append(original_video.subclipped(start_time, end_time))
            final_audio_segments.append(bg_audio_full[start_time_ms:end_time_ms])

        last_pos = end_time
        progress_bar.progress((i + 1) / total_segments, text=f"Processing segment {i+1}/{total_segments}")

    # --- 3. Handle End Gap ---
    if last_pos < original_video.duration:
         final_video_clips.append(original_video.subclipped(last_pos, original_video.duration))
         final_audio_segments.append(bg_audio_full[int(last_pos*1000):])

    # --- 4. Final Concatenation ---
    logger.log("🧩 Concatenating video and audio segments...")
    
    # Concatenate Audio
    final_audio_track = sum(final_audio_segments, AudioSegment.empty())
    final_audio_path = os.path.join(output_dir, f"{base_name}_dubbed_audio.wav")
    final_audio_track.export(final_audio_path, format="wav")
    
    # Concatenate Video
    final_video_clip = concatenate_videoclips(final_video_clips)
    
    # Merge
    final_video_path = os.path.join(output_dir, f"{base_name}_dubbed_video.mp4")
    logger.log(f"🎬 Merging final output...")
    
    # We write the video file directly with the new audio
    final_video_clip.write_videofile(
        final_video_path, 
        codec="libx264", 
        audio=final_audio_path, 
        audio_codec="aac", 
        logger=None
    )
    
    original_video.close()
    logger.log(f"✅ Final video saved successfully to '{os.path.basename(final_video_path)}'")
    
    # Upload final video back to GCS
    if config.get('BUCKET_NAME'):
        destination_blob = f"output_dubbed_videos/{os.path.basename(final_video_path)}"
        gcs_uri = upload_blob(get_gcs_client(), config['BUCKET_NAME'], final_video_path, destination_blob)
        if gcs_uri:
            logger.log(f"✅ Final video uploaded to GCS: {gcs_uri}")
    
    return final_video_path


def main():
    st.set_page_config(layout="wide", page_title="Gemini Video Dubber")
    st.title("🎬 Gemini Video Dubbing Studio")
    st.markdown("Upload a video, analyze it to generate a script, refine voice assignments, and create your dubbed video.")

    # --- Initialize Session State ---
    if 'processing_stage' not in st.session_state:
        st.session_state['processing_stage'] = 0 # 0: Start, 1: Script Generated, 2: Completed
    if 'dubbing_script' not in st.session_state:
        st.session_state['dubbing_script'] = None
    if 'speaker_assignments' not in st.session_state:
        st.session_state['speaker_assignments'] = []
    if 'local_video_path' not in st.session_state:
        st.session_state['local_video_path'] = None
    if 'background_track_path' not in st.session_state:
        st.session_state['background_track_path'] = None
    if 'output_dir' not in st.session_state:
        st.session_state['output_dir'] = None

    # --- Sidebar for Inputs ---
    with st.sidebar:
        st.header("🛠️ Configuration")
        
        # --- GCS Integration UI ---
        st.subheader("📁 File Selection")
        
        gcs_client = None
        buckets = []
        
        use_gcs = st.checkbox("Enable Google Cloud Storage (GCS)")
        
        if use_gcs:
            try:
                gcs_client = get_gcs_client()
                buckets = list_gcs_buckets(gcs_client)
            except Exception as e:
                st.warning(f"⚠️ Could not initialize GCS: {e}")
        
        if use_gcs and buckets:
             selected_bucket = st.selectbox("Select GCS Bucket", options=buckets)
        else:
             selected_bucket = None

        input_method = st.radio("Input Method", ["Select from GCS", "Upload Local File"])
        
        gcs_file_path = None
        uploaded_file = None
        
        if input_method == "Select from GCS":
            if selected_bucket and gcs_client:
                files = list_gcs_files(gcs_client, selected_bucket)
                gcs_file_path = st.selectbox("Select Video File", options=files)
        else:
            uploaded_file = st.file_uploader("Upload Input Video", type=["mp4", "mov", "avi", "mkv"])

        st.markdown("---")

        with st.form("config_form"):
            google_api_key = st.text_input("Google API Key", type="password", help="Required if not using Vertex AI.")
            use_vertex_ai = st.checkbox("Use Vertex AI", value=True)
            
            project_id = None
            location = None
            if use_vertex_ai:
                project_id = st.text_input("Google Cloud Project ID", help="Required for Vertex AI.")
                location = st.selectbox("GCP Location", options=GCP_REGIONS, index=0)

            col1, col2 = st.columns(2)
            with col1:
                input_lang = st.selectbox("Input Language", options=LANGUAGES, index=LANGUAGES.index("Chinese"))
            with col2:
                output_lang = st.selectbox("Output Language", options=LANGUAGES, index=LANGUAGES.index("Hindi"))

            llm_model_name = st.selectbox("LLM Model Name", options=["gemini-3-pro-preview", "gemini-3-flash-preview"], index=0)
            tts_model_name = st.selectbox("TTS Model Name (Vertex AI)", options=["gemini-2.5-pro-tts", "gemini-2.5-flash-tts"], index=0)
            
            # Action Buttons in Form
            analyze_clicked = st.form_submit_button("1. Analyze Video & Generate Script")

    # --- Main Logic ---
    logger = StatusLogger()

    # Step 1: Analysis Trigger
    if analyze_clicked:
        # Validation
        if not use_vertex_ai and not google_api_key:
            st.error("Please provide a Google API Key or select 'Use Vertex AI'.")
        elif use_vertex_ai and (not project_id or not location):
            st.error("Please provide a Project ID and Location for Vertex AI.")
        elif (input_method == "Select from GCS" and not gcs_file_path) or \
             (input_method == "Upload Local File" and not uploaded_file):
            st.error("Please select or upload a video file.")
        else:
            # Prepare Files
            st.session_state['processing_stage'] = 0 # Reset
            
            # We need to save the file to a permanent location for this session
            # to avoid TempDirectory cleanup issues between clicks.
            session_work_dir = os.path.join("/Users/shashirr/Shashi/scvideos", "session_workspace")
            os.makedirs(session_work_dir, exist_ok=True)
            
            local_processing_path = None
            gcs_video_uri = None

            if input_method == "Upload Local File":
                st.info("Uploading file to GCS for efficient processing...")
                local_temp_path = os.path.join(session_work_dir, uploaded_file.name)
                with open(local_temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Upload to GCS
                destination_blob = f"uploads/{uploaded_file.name}"
                gcs_video_uri = upload_blob(gcs_client, selected_bucket, local_temp_path, destination_blob)
                local_processing_path = local_temp_path
                
            else: # GCS
                st.info("Downloading file from GCS for local audio processing...")
                gcs_video_uri = f"gs://{selected_bucket}/{gcs_file_path}"
                local_temp_path = os.path.join(session_work_dir, os.path.basename(gcs_file_path))
                download_success = download_blob(gcs_client, selected_bucket, gcs_file_path, local_temp_path)
                if download_success:
                    local_processing_path = local_temp_path
                else:
                    st.error("Failed to download file from GCS.")
                    st.stop()
            
            st.session_state['local_video_path'] = local_processing_path

            final_llm_model = llm_model_name
            if not use_vertex_ai:
                final_llm_model = f"models/{llm_model_name}"

            config = {
                "GOOGLE_API_KEY": google_api_key,
                "USE_VERTEX_AI": use_vertex_ai,
                "PROJECT_ID": project_id,
                "LOCATION": location,
                "INPUT_LANGUAGE": input_lang,
                "OUTPUT_LANGUAGE": output_lang,
                "MODEL_NAME": final_llm_model,
                "TTS_MODEL": tts_model_name,
                "BUCKET_NAME": selected_bucket
            }
            
            st.info("Running Step 1: Script Generation...")
            dubbing_script, background_path, output_dir = step1_generate_script(local_processing_path, gcs_video_uri, config, logger)
            
            if dubbing_script:
                st.session_state['dubbing_script'] = dubbing_script
                st.session_state['background_track_path'] = background_path
                st.session_state['output_dir'] = output_dir
                st.session_state['processing_stage'] = 1
                
                # Initial Assignment
                initial_assignments = assign_specific_voices(dubbing_script)
                st.session_state['speaker_assignments'] = initial_assignments
                st.rerun()
            else:
                st.error("Script generation failed.")

    # --- Step 2: Review and Synthesize ---
    if st.session_state['processing_stage'] >= 1:
        st.divider()
        st.header("📝 Review Script & 🎙️ Assign Voices")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.expander("View Generated Script", expanded=False):
                st.json(st.session_state['dubbing_script'])

        with col2:
            st.subheader("Voice Assignments")
            st.info("Edit the voice assigned to each speaker below.")
            
            # Editable Table for Voices
            # We create a list of voices combined (Male + Female) for the dropdown
            all_voices = sorted(list(set(MALE_VOICE_LIST + FEMALE_VOICE_LIST + CHILD_VOICE_LIST)))
            
            updated_assignments = []
            
            for i, assignment in enumerate(st.session_state['speaker_assignments']):
                st.markdown(f"**{assignment['speaker_label']}** ({assignment['character_type']})")
                
                # Determine index for default selection
                current_voice = assignment['selected_voice']
                try:
                    default_idx = all_voices.index(current_voice)
                except ValueError:
                    default_idx = 0
                
                new_voice = st.selectbox(
                    f"Voice for {assignment['speaker_label']}", 
                    options=all_voices, 
                    index=default_idx, 
                    key=f"voice_select_{i}"
                )
                
                updated_assignments.append({
                    "speaker_label": assignment['speaker_label'],
                    "character_type": assignment['character_type'],
                    "selected_voice": new_voice
                })
            
            # Update session state with user choices
            st.session_state['speaker_assignments'] = updated_assignments

        st.divider()
        synthesize_clicked = st.button("2. Synthesize & Dub Video")
        
        if synthesize_clicked:
            # Re-construct config from session/inputs is tricky if inputs cleared. 
            # Ideally we should store config in session state too.
            # For now, we assume user hasn't cleared the form inputs (Streamlit keeps them usually).
            # But safer to reconstruct minimal config needed for Step 2.
            
            # Need to grab keys again from the form (which might not be submitted this run)
            # Streamlit widgets retain state.
            pass # proceed below

    if st.session_state['processing_stage'] >= 1 and 'synthesize_clicked' in locals() and synthesize_clicked:
         # Pack config again
        final_llm_model = llm_model_name
        if not use_vertex_ai:
            final_llm_model = f"models/{llm_model_name}"

        config = {
            "GOOGLE_API_KEY": google_api_key,
            "USE_VERTEX_AI": use_vertex_ai,
            "PROJECT_ID": project_id,
            "LOCATION": location,
            "INPUT_LANGUAGE": input_lang,
            "OUTPUT_LANGUAGE": output_lang,
            "MODEL_NAME": final_llm_model,
            "TTS_MODEL": tts_model_name,
            "BUCKET_NAME": selected_bucket
        }
        
        final_video = step2_synthesize_video(
            st.session_state['local_video_path'],
            st.session_state['dubbing_script'],
            st.session_state['speaker_assignments'],
            st.session_state['background_track_path'],
            st.session_state['output_dir'],
            config,
            logger
        )
        
        if final_video and os.path.exists(final_video):
            st.session_state['processing_stage'] = 2
            st.balloons()
            st.header("🎉 Dubbing Complete!")
            
            with open(final_video, 'rb') as vf:
                st.video(vf.read())
                
            with open(final_video, 'rb') as vf:
                st.download_button(
                    label="Download Dubbed Video",
                    data=vf,
                    file_name=os.path.basename(final_video),
                    mime="video/mp4"
                )


if __name__ == "__main__":
    main()