import os
import streamlit as st
import google.genai as genai
import time
import json
from moviepy import VideoFileClip, AudioFileClip
from pydub import AudioSegment
import subprocess
from google.genai import types
import wave
import tempfile
import datetime
from google.cloud import storage
import pyrubberband as rb
import soundfile as sf
import concurrent.futures
import queue

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
MALE_VOICE_LIST = ['Puck', 'Orus', 'Enceladus', 'Charon', 'Fenrir', 'Iapetus', 'Umbriel']
FEMALE_VOICE_LIST = ['Kore', 'Zephyr', 'Leda', 'Sulafat', 'Aoede', 'Callirrhoe', 'Autonoe']
CHILD_VOICE_LIST = ['Leda', 'Kore']
FALLBACK_VOICE = 'Leda'

# --- DEFAULT PROMPT ---
# This is the default prompt that will be shown in the UI for editing.
DEFAULT_VIDEO_ANALYSIS_PROMPT = """
You are an expert voice director and video producer creating a script for dubbing.
Analyze the provided video file's audio track with extreme detail. Your goal is to capture the complete performance, including the emotional and dramatic context.

Follow these steps precisely:
1.  **Speaker Diarization**: Use a combination of audio and video to identify every distinct speaker and assign a unique label (e.g., SPEAKER_1).
2.  **Character Classification**: Classify each speaker as MALE, FEMALE, or CHILD.
3.  **Emotional Analysis**: For each dialogue segment, identify the primary emotion being conveyed. Choose from this list: **Love & Affection: PASSIONATE, LONGING, ADORING, FLIRTATIOUS, TENDER, SHY
                  Joy & Happiness: ELATED, AMUSED, CONTENT, RELIEVED, HOPEFUL
                  Anger & Fury: IRRITATED, FRUSTRATED, RAGING, INDIGNANT, VENGEFUL, CONTEMPTUOUS
                  Sadness & Grief: SORROWFUL, HEARTBROKEN, DESPAIRING, MELANCHOLIC, SYMPATHETIC
                  Fear & Anxiety: TERRIFIED, ANXIOUS, NERVOUS, DREADFUL, PANICKED
                  Surprise & Wonder: SHOCKED, ASTONISHED, AWESTRUCK, DISBELIEF
                  Complex & Social: GUILTY, ASHAMED, JEALOUS, BETRAYED, DESPERATE, ARROGANT, SUSPICIOUS
                  Neutral: NEUTRAL**.
4.  **Delivery Style Analysis**: For each dialogue segment, identify the style of delivery. Choose from this list: **NORMAL, SHOUTING, WHISPERING, PLEADING, CRYING / SOBBING, LAUGHING, MOCKING / SARCASTIC,
                         MENACING, FRANTIC, HESITANT, FIRM**.
5.  **Transcription & Translation**: Provide the timestamped original '{INPUT_LANGUAGE}' transcript and its accurate and meaningful '{OUTPUT_LANGUAGE}' translation as used in movies.
6.  **Pace of Speech**: For each dialogue segment, identify the pace of delivery. Choose from this list: **NORMAL, FAST, VERY FAST, SLOW, MEDIUM, VERY SLOW**.
7.  **Time Conversion**: Always assign the start and end time in seconds. Do not consider minutes. For example, if a time shows as 1:12, it should be converted to 72 seconds (60+12 seconds) and not 112 seconds.
8.  **Non-Dialogue Sounds**: Capture any significant non-dialogue vocal sounds like sighs, gasps, laughs, or cries and include them in the transcript.
9.  **Output Format**: Your final output MUST be a valid JSON array of objects. Do not include any text or explanations outside of this array. Each object represents a single line of dialogue and must have the following structure:
    {{
          "start_time": float,
          "end_time": float,
          "speaker_label": "string",
          "character_type": "string (MALE, FEMALE, or CHILD)",
          "emotion": "string (e.g., Love & Affection: PASSIONATE, LONGING, ADORING, FLIRTATIOUS, TENDER, SHY
                      Joy & Happiness: ELATED, AMUSED, CONTENT, RELIEVED, HOPEFUL
                      Anger & Fury: IRRITATED, FRUSTRATED, RAGING, INDIGNANT, VENGEFUL, CONTEMPTUOUS
                      Sadness & Grief: SORROWFUL, HEARTBROKEN, DESPAIRING, MELANCHOLIC, SYMPATHETIC
                      Fear & Anxiety: TERRIFIED, ANXIOUS, NERVOUS, DREADFUL, PANICKED
                      Surprise & Wonder: SHOCKED, ASTONISHED, AWESTRUCK, DISBELIEF
                      Complex & Social: GUILTY, ASHAMED, JEALOUS, BETRAYED, DESPERATE, ARROGANT, SUSPICIOUS
                      Neutral: NEUTRAL)",
          "delivery_style": "string (NORMAL, SHOUTING, WHISPERING, PLEADING, CRYING / SOBBING, LAUGHING, MOCKING / SARCASTIC,
                             MENACING, FRANTIC, HESITANT, FIRM)",
          "original_transcript": "string ('{INPUT_LANGUAGE}' text)",
          "{OUTPUT_LANGUAGE}_translation": "string ('{OUTPUT_LANGUAGE}' text)",
          "pace": "string (NORMAL, FAST, VERY FAST, SLOW, MEDIUM or VERY SLOW)"
    }}
"""

@st.cache_resource
def get_gcs_client():
    """Get GCS client, cached for performance."""
    return storage.Client()

def list_gcs_buckets(client):
    """Lists all GCS buckets."""
    try:
        return [bucket.name for bucket in client.list_buckets()]
    except Exception as e:
        st.error(f"Failed to list GCS buckets. Please ensure permissions are correct. Error: {e}")
        return []

def list_gcs_dirs_files(client, bucket_name, prefix):
    """A more robust function to list directories and files."""
    dirs = set()
    files = []
    try:
        if prefix and not prefix.endswith('/'):
            prefix += '/'

        all_blobs = client.list_blobs(bucket_name, prefix=prefix)
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv']

        for blob in all_blobs:
            if blob.name == prefix:
                continue
            relative_path = blob.name[len(prefix):]
            if '/' in relative_path:
                top_level_dir = relative_path.split('/')[0]
                dirs.add(top_level_dir)
            else:
                if any(relative_path.lower().endswith(ext) for ext in video_extensions):
                    files.append(blob.name)
        
        full_path_dirs = [f"{prefix}{d}/" for d in sorted(list(dirs))]
        return full_path_dirs, sorted(files)

    except Exception as e:
        st.error(f"❌ Error listing content in gs://{bucket_name}/{prefix or ''}. Error: {e}")
        return [], []

def download_gcs_file(client, bucket_name, source_blob_name, logger):
    """Downloads a blob from the bucket to a local temporary file."""
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(source_blob_name)[1]) as tmpfile:
            logger.log(f"⬇️ Downloading gs://{bucket_name}/{source_blob_name}...")
            blob.download_to_filename(tmpfile.name)
            logger.log("✅ Download complete.")
            return tmpfile.name
    except Exception as e:
        logger.log(f"❌ Failed to download file from GCS: {e}")
        st.error(f"Failed to download gs://{bucket_name}/{source_blob_name}. Error: {e}")
        return None

def upload_to_gcs(client, bucket_name, source_file_path, destination_blob_name, logger):
    """Uploads a file to the bucket."""
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        logger.log(f"⬆️ Uploading final video to gs://{bucket_name}/{destination_blob_name}...")
        blob.upload_from_filename(source_file_path)
        logger.log("✅ Upload complete.")
        return True
    except Exception as e:
        logger.log(f"❌ Failed to upload file to GCS: {e}")
        st.error(f"Failed to upload to gs://{bucket_name}/{destination_blob_name}. Error: {e}")
        return False

def generate_download_signed_url_v4(client, bucket_name, blob_name):
    """Generates a v4 signed URL for downloading a blob."""
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(hours=1),
            method="GET",
        )
        return url
    except Exception as e:
        st.error(f"Could not generate download link. Error: {e}")
        return None

class StatusLogger:
    """A thread-safe logger that puts messages onto a queue."""
    def __init__(self, log_queue: queue.Queue):
        self.queue = log_queue
    
    def log(self, message: str):
        """Puts a message onto the log queue."""
        self.queue.put(message)

def render_logs(container, current_logs: list):
    """Renders a list of log messages into a Streamlit container."""
    with container:
        st.subheader("⚙️ Processing Status")
        for msg in reversed(current_logs):
            timestamp_str, _, message_body = msg.partition(" - ")
            if "✅" in message_body or "🎉" in message_body:
                st.success(msg)
            elif "❌" in message_body or "⚠️" in message_body:
                st.warning(msg)
            else:
                st.info(msg)

# --- CORE LOGIC ---

def get_dubbing_script_from_video(video_path, config, logger):
    """
    Uploads a video to the Gemini API and analyzes it using a provided prompt.
    """
    try:
        if config["USE_VERTEX_AI"]:
            logger.log(f"Authenticating with Vertex AI (Project: {config['PROJECT_ID']}, Location: {config['LOCATION']})")
            client = genai.Client(project=config["PROJECT_ID"], location=config["LOCATION"])
        else:
            logger.log("Authenticating with Google API Key")
            client = genai.Client(api_key=config["GOOGLE_API_KEY"])
    except Exception as e:
        logger.log(f"❌ Authentication failed: {e}")
        return None

    logger.log(f"Uploading video '{os.path.basename(video_path)}' to the Gemini API...")
    video_file = client.files.upload(file=video_path)
    
    logger.log("...Video is processing on the server")
    while video_file.state.name == "PROCESSING":
        time.sleep(10)
        video_file = client.files.get(name=video_file.name)

    if video_file.state.name == "FAILED":
        logger.log(f"❌ Video processing failed: {video_file.state}")
        raise ValueError(f"Video processing failed: {video_file.state}")

    logger.log(f"✅ Video uploaded and processed successfully: {video_file.name}")

    # <<< MODIFIED: Get the prompt from the config and format it >>>
    raw_prompt = config['VIDEO_ANALYSIS_PROMPT']
    prompt = raw_prompt.format(
        INPUT_LANGUAGE=config["INPUT_LANGUAGE"],
        OUTPUT_LANGUAGE=config["OUTPUT_LANGUAGE"]
    )
    
    logger.log(f"🤖 Sending request to {config['MODEL_NAME']} for analysis...")
    response = client.models.generate_content(
        model=config['MODEL_NAME'],
        contents=[video_file, "\n\n", prompt]
    )
    client.files.delete(name=video_file.name)
    logger.log(f"Cleaned up uploaded file on server.")

    try:
        json_text = response.text.strip().lstrip("```json").rstrip("```")
        dubbing_script = json.loads(json_text)
        dubbing_script.sort(key=lambda x: x['start_time'])
        logger.log("✅ Successfully received and parsed the dubbing script from Gemini.")
        #with st.expander("Show Generated Dubbing Script (JSON)"):
        #    st.json(dubbing_script)
        return dubbing_script
    except (json.JSONDecodeError, IndexError, AttributeError) as e:
        logger.log(f"❌ Failed to parse JSON from Gemini's response: {e}")
        logger.log(f"RAW RESPONSE:\n{response.text}")
        return None

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
    logger.log("🎶 Separating background music with Demucs...")
    try:
        command = ["python3", "-m", "demucs.separate", "-n", "htdemucs", "-o", str(output_dir), "--two-stems", "vocals", str(audio_path)]
        #with st.spinner('Demucs is separating audio tracks... This may take some time.'):
        subprocess.run(command, check=True, capture_output=True, text=True)
        
        audio_filename = os.path.splitext(os.path.basename(audio_path))[0]
        background_path = os.path.join(output_dir, "htdemucs", audio_filename, "no_vocals.wav")

        if os.path.exists(background_path):
            logger.log("✅ Background music separated successfully.")
            return background_path
        else:
            logger.log("❌ Demucs did not produce the background music file.")
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
            client = genai.Client(project=config["PROJECT_ID"], location=config["LOCATION"])
        else:
            client = genai.Client(api_key=config["GOOGLE_API_KEY"])
    except Exception as e:
        logger.log(f"❌ TTS Authentication failed: {e}")
        return None, ""
    
    # Construct TTS prompt
    voice_description = f"a standard '{config['OUTPUT_LANGUAGE']}' {segment_details['character_type'].lower()} voice"
    style_description = f" in a {segment_details['emotion'].lower()}, {segment_details['delivery_style'].lower()} tone"
    instruction = f"You are a movie dubbing expert. Maintain the duration of the output audio within {segment_details['clip_duration']} milliseconds."
    full_prompt = f"{instruction}. Using {voice_description}{style_description}, {segment_details['speaker_label']} says the following at {segment_details['pace']} speed: {text}"

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
            )
        )

        if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
            logger.log(f"❌ Gemini returned an empty response for the text. Skipping.")
            return None, full_prompt

        audio_data = response.candidates[0].content.parts[0].inline_data.data
        if not audio_data:
            logger.log(f"❌ Gemini did not return audio data for the text.")
            return None, full_prompt

        wave_file(segment_details['output_path'], audio_data)
        return segment_details['output_path'], full_prompt
    except Exception as e:
        logger.log(f"❌ Error synthesizing speech with Gemini: {e}")
        return None, full_prompt

def merge_audio_with_video(video_path, audio_path, output_path, logger):
    logger.log(f"🎬 Merging final audio with video...")
    try:
        with VideoFileClip(video_path) as video_clip, AudioFileClip(audio_path) as audio_clip:
            video_clip.audio = audio_clip
            video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
        logger.log(f"🎉 Final video saved successfully to '{os.path.basename(output_path)}'")
        return output_path
    except Exception as e:
        logger.log(f"❌ An error occurred during video merging: {e}")
        return None

def assign_specific_voices(transcript_data):
    speaker_info = {item['speaker_label']: item['character_type'] for item in transcript_data if item['speaker_label'] not in {}}
    voice_indices = {'MALE': 0, 'FEMALE': 0, 'CHILD': 0}
    speaker_voice_array = []
    for speaker in sorted(speaker_info.keys()):
        char_type = speaker_info[speaker]
        voice_list = globals().get(f"{char_type}_VOICE_LIST", [])
        if voice_list:
            index = voice_indices[char_type] % len(voice_list)
            selected_voice = voice_list[index]
            voice_indices[char_type] += 1
        else:
            selected_voice = FALLBACK_VOICE
        speaker_voice_array.append({"speaker_label": speaker, "character_type": char_type, "selected_voice": selected_voice})
    return speaker_voice_array

def process_video_dubbing(video_path, config, logger, log_container, synthesis_log_area):
    """Main function to orchestrate the entire dubbing process."""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, f"output_{base_name}")
        os.makedirs(output_dir, exist_ok=True)
        
        original_audio_path = os.path.join(output_dir, f"{base_name}_original_audio.wav")
        extract_audio(video_path, original_audio_path, logger)
        # --- MODIFIED: Run Demucs and Gemini analysis in parallel ---
        background_track_path = None
        dubbing_script = None

                # --- NEW: Polling loop for live logging ---
        with concurrent.futures.ThreadPoolExecutor() as executor:
            logger.log("🚀 Starting parallel processing for audio separation and video analysis...")
            render_logs(log_container, st.session_state.log_messages)
            demucs_output_dir = os.path.join(output_dir, "separated")
            future_audio = executor.submit(separate_background_music, original_audio_path, demucs_output_dir, logger)
            future_script = executor.submit(get_dubbing_script_from_video, video_path, config, logger)
            
            futures = [future_audio, future_script]
            while any(not f.done() for f in futures):
                while not logger.queue.empty():
                    message = logger.queue.get_nowait()
                    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S")
                    st.session_state.log_messages.append(f"{timestamp} UTC - {message}")
                render_logs(log_container, st.session_state.log_messages)
                time.sleep(0.5)
            
            dubbing_script = future_script.result()
            background_track_path = future_audio.result()

        # Final log render to catch any remaining messages
        while not logger.queue.empty():
            message = logger.queue.get_nowait()
            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S")
            st.session_state.log_messages.append(f"{timestamp} UTC - {message}")
        render_logs(log_container, st.session_state.log_messages)
        # --- END NEW ---

        logger.log("✅ Parallel processing finished.")
        
        if not dubbing_script:
            logger.log("❌ Aborting due to failure in script generation.")
            return None
 
        # >>>>> CHANGE: ADDED st.expander HERE IN THE MAIN THREAD <<<<<
        with st.expander("Show Generated Dubbing Script (JSON)", expanded=True):
            st.json(dubbing_script)

        logger.log(f"🔊 Initializing {config['TTS_MODEL']} for Text-to-Speech...")
        
        with VideoFileClip(video_path) as clip:
            video_duration_ms = int(clip.duration * 1000)

        if background_track_path and os.path.exists(background_track_path):
            background_music = AudioSegment.from_wav(background_track_path)
            if len(background_music) < video_duration_ms:
                 background_music += AudioSegment.silent(duration=video_duration_ms - len(background_music))
        else:
            logger.log("⚠️ Could not separate background music. Using a silent background.")
            background_music = AudioSegment.silent(duration=video_duration_ms)

        final_vocal_track = AudioSegment.silent(duration=len(background_music))
        speaker_assignments = assign_specific_voices(dubbing_script)
        logger.log("Assigned voices to speakers.")

        output_lang_key = f"{config['OUTPUT_LANGUAGE']}_translation"
        progress_bar = st.progress(0, text="Synthesizing audio segments...")

        for i, segment in enumerate(dubbing_script):
            output_text = segment.get(output_lang_key, "...")
            start_time_ms = int(segment['start_time'] * 1000)
            end_time_ms = int(segment['end_time'] * 1000)
            
            selected_voice = next((item['selected_voice'] for item in speaker_assignments if item['speaker_label'] == segment['speaker_label']), FALLBACK_VOICE)
            
            segment_details = {
                'character_type': segment['character_type'], 'emotion': segment.get('emotion', 'NEUTRAL'),
                'delivery_style': segment.get('delivery_style', 'NORMAL'), 'speaker_label': segment.get('speaker_label', 'DEFAULT'),
                'pace': segment.get('pace', 'NORMAL'), 'clip_duration': end_time_ms - start_time_ms,
                'selected_voice': selected_voice, 'output_path': os.path.join(output_dir, f"segment_{i}.wav")
            }
            
            time.sleep(2)
            synthesized_path, tts_prompt = synthesize_speech_with_gemini(output_text, segment_details, config, logger)
            
            with synthesis_log_area.expander(f"Segment {i+1}: Speaker - {segment.get('speaker_label', 'N/A')} ({segment['start_time']:.2f}s - {segment['end_time']:.2f}s)"):
                st.markdown(f"**🗣️ Translated Text:** `{output_text}`")
                st.markdown(f"**🤖 TTS Prompt:** `{tts_prompt}`")
                if synthesized_path and os.path.exists(synthesized_path):
                    st.markdown("**🔊 Generated Audio:**")
                    with open(synthesized_path, "rb") as audio_file:
                        st.audio(audio_file.read(), format="audio/wav")
                else:
                    st.warning("Audio synthesis failed for this segment.")

            if synthesized_path and os.path.exists(synthesized_path):
                with open(synthesized_path, "rb") as f:
                    dub_segment = AudioSegment.from_wav(f)
                                # --- NEW: AUDIO SPEED ADJUSTMENT LOGIC ---
                    try:
                      original_duration_ms = len(dub_segment)
                      target_duration_ms = segment_details['clip_duration']
                    
                      if target_duration_ms > 0 and original_duration_ms > 0:
                        speed_ratio = original_duration_ms / target_duration_ms
                        if speed_ratio > 1.5:
                            speed_ratio = 1.27
                        
                        # Only adjust if the speed difference is significant (e.g., > 5%)
                        if abs(1 - speed_ratio) > 0.05:
                            logger.log(f"   ⏱️ Adjusting speed for segment {i}. Ratio: {speed_ratio:.2f} (Original: {original_duration_ms}ms, Target: {target_duration_ms}ms)")
                            
                            stretched_audio_path = os.path.join(output_dir, f"segment_{i}_stretched.wav")
                            
                            # Read audio data, stretch it with pyrubberband, and write to a new file
                            y, sr = sf.read(synthesized_path)
                            y_stretched = rb.time_stretch(y, sr, speed_ratio)
                            sf.write(stretched_audio_path, y_stretched, sr)

                            # Load the new, time-adjusted audio segment
                            dub_segment = AudioSegment.from_wav(stretched_audio_path)
                            #os.remove(stretched_audio_path) # Clean up intermediate file
                    except Exception as e:
                      logger.log(f"   ⚠️ Could not time-stretch segment {i}: {e}")
                
                final_vocal_track = final_vocal_track.overlay(dub_segment, position=start_time_ms)
                os.remove(synthesized_path)
            else:
                logger.log(f"⚠️ Segment {i} could not be synthesized and will be silent.")
            
            progress_bar.progress((i + 1) / len(dubbing_script), text=f"Synthesizing audio segment {i+1}/{len(dubbing_script)}")
        
        logger.log("🎤 Speech synthesis complete. Combining audio tracks...")
        final_audio_track = background_music.overlay(final_vocal_track)
        final_audio_path = os.path.join(output_dir, f"{base_name}_dubbed_audio.wav")
        final_audio_track.export(final_audio_path, format="wav")
        logger.log("✅ Final audio track created.")

        final_video_path = os.path.join(output_dir, f"dubbed_{base_name}.mp4")
        merged_video_path = merge_audio_with_video(video_path, final_audio_path, final_video_path, logger)
        
        if merged_video_path and os.path.exists(merged_video_path):
            gcs_client = get_gcs_client()
            destination_blob_name = f"dubbed_videos/{os.path.basename(merged_video_path)}"
            if upload_to_gcs(gcs_client, config['BUCKET_NAME'], merged_video_path, destination_blob_name, logger):
                return destination_blob_name
        
        return None

def handle_navigation():
    """Callback to update path based on folder navigation."""
    nav_choice = st.session_state.get("gcs_nav_choice")
    if not nav_choice: return
    if nav_choice == "⬆️ Go Up (..)":
        path_parts = st.session_state.current_path.strip('/').split('/')
        st.session_state.current_path = '/'.join(path_parts[:-1]) + '/' if len(path_parts) > 1 else ""
    else:
        st.session_state.current_path = nav_choice
    st.session_state.gcs_file_path = None
    st.session_state.gcs_file_choice = None
    st.session_state.gcs_nav_choice = None

# --- STREAMLIT UI ---
def main():
    st.set_page_config(layout="wide", page_title="Gemini Video Dubber")
    st.title("🎬 Gemini Video Dubbing Studio")
    st.markdown("Select a video from GCS, configure the settings, and let Gemini automatically dub it for you.")

# >>>>> CHANGE: INITIALIZE SESSION STATE AT THE TOP OF THE SCRIPT RUN <<<<<
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []

    gcs_client = get_gcs_client()

    # Initialize session state
    for key, default_val in [('current_path', ""), ('selected_bucket', None), 
                             ('gcs_file_path', None), ('gcs_nav_choice', None), 
                             ('gcs_file_choice', None)]:
        if key not in st.session_state:
            st.session_state[key] = default_val

    def reset_on_bucket_change():
        st.session_state.current_path = ""
        st.session_state.gcs_file_path = None
        st.session_state.gcs_nav_choice = None 
        st.session_state.gcs_file_choice = None

    def handle_file_selection():
        st.session_state.gcs_file_path = st.session_state.get("gcs_file_choice")

    # --- Sidebar UI ---
    with st.sidebar:
        st.header("🛠️ Configuration")
        st.subheader("📁 Select Video from GCS")

        buckets = list_gcs_buckets(gcs_client)
        st.selectbox("1. Select GCS Bucket", options=[""] + buckets, key="selected_bucket", on_change=reset_on_bucket_change)

        if st.session_state.selected_bucket:
            st.markdown("---")
            st.text_input("📍 Current Location:", value=st.session_state.current_path or "Bucket Root", disabled=True)

            dirs, files = list_gcs_dirs_files(gcs_client, st.session_state.selected_bucket, st.session_state.current_path)
            nav_options = ["⬆️ Go Up (..)"] + dirs if st.session_state.current_path else dirs
            st.selectbox("2. Navigate Folders:", options=nav_options, key="gcs_nav_choice", on_change=handle_navigation,
                         format_func=lambda p: "⬆️ Go Up (..)" if ".." in p else f"📁 {p.rstrip('/').split('/')[-1]}", placeholder="Select a folder to enter...")
            st.selectbox("3. Select Video File:", options=[""] + files, key="gcs_file_choice", on_change=handle_file_selection,
                         format_func=lambda p: p.split('/')[-1] if p else "", placeholder="Select a video file...")
            st.markdown("---")

        with st.form("config_form"):
            st.subheader("🔑 API & Model Settings")
            google_api_key = st.text_input("Google API Key", type="password", help="Required if not using Vertex AI.")
            use_vertex_ai = st.checkbox("Use Vertex AI", value=False)
            project_id, location = None, None
            if use_vertex_ai:
                project_id = st.text_input("Google Cloud Project ID", help="Required for Vertex AI.")
                location = st.selectbox("GCP Location", options=GCP_REGIONS, index=0)

            st.subheader("⚙️ Dubbing Settings")
            col1, col2 = st.columns(2)
            with col1:
                input_lang = st.selectbox("Input Language", options=LANGUAGES, index=LANGUAGES.index("Chinese"))
            with col2:
                output_lang = st.selectbox("Output Language", options=LANGUAGES, index=LANGUAGES.index("Hindi"))

            llm_model_name = st.selectbox("LLM Model Name", options=["gemini-2.5-pro", "gemini-2.5-flash"], index=0)
            tts_model_name = st.selectbox("TTS Model Name", options=["gemini-2.5-pro-preview-tts", "gemini-2.5-flash-preview-tts"], index=0)
            
            # <<< MODIFIED: Added editable prompt text area >>>
            st.subheader("🤖 Video Analysis Prompt")
            with st.expander("Edit the prompt for video analysis"):
                video_prompt_text = st.text_area(
                    "Prompt:", 
                    value=DEFAULT_VIDEO_ANALYSIS_PROMPT, 
                    height=400,
                    key="editable_video_prompt"
                )

            submitted = st.form_submit_button("🚀 Start Dubbing Process")

    # --- Main Page Logic ---
    if submitted:
        if not use_vertex_ai and not google_api_key:
            st.error("Please provide a Google API Key or select 'Use Vertex AI'.")
        elif use_vertex_ai and (not project_id or not location):
            st.error("Please provide a Project ID and Location for Vertex AI.")
        elif not st.session_state.gcs_file_path or not st.session_state.selected_bucket:
            st.error("Please select a bucket and a video file from GCS.")
        else:
            # Create placeholders for dynamic content
            log_container = st.empty()
            main_content_area = st.container()
            
            with main_content_area:
                st.subheader("🔬 Detailed Synthesis Log")
                synthesis_log_area = st.container(height=600, border=True)
            
# --- CHANGE: Use the queue-based logger ---
            st.session_state.log_messages = []
            log_queue = queue.Queue()
            logger = StatusLogger(log_queue)

            render_logs(log_container, st.session_state.log_messages)

            logger.log("🚀 Initializing process...")

            local_video_path = download_gcs_file(gcs_client, st.session_state.selected_bucket, st.session_state.gcs_file_path, logger)
            
            if local_video_path:
                config = {
                    "GOOGLE_API_KEY": google_api_key, "USE_VERTEX_AI": use_vertex_ai,
                    "PROJECT_ID": project_id, "LOCATION": location, "INPUT_LANGUAGE": input_lang,
                    "OUTPUT_LANGUAGE": output_lang, "MODEL_NAME": f"models/{llm_model_name}",
                    "TTS_MODEL": tts_model_name, "BUCKET_NAME": st.session_state.selected_bucket,
                    "VIDEO_ANALYSIS_PROMPT": video_prompt_text # <<< MODIFIED: Pass edited prompt
                }
                
                st.info(f"Starting dubbing process for **gs://{config['BUCKET_NAME']}/{st.session_state.gcs_file_path}**...")
                final_gcs_path = None
                try:
                    final_gcs_path = process_video_dubbing(local_video_path, config, logger, log_container, synthesis_log_area)
                except Exception as e:
                    logger.log(f"❌ An unexpected error occurred in the main process: {e}")
                    st.exception(e)
                finally:
                    logger.log(f"🧹 Cleaning up temporary file: {os.path.basename(local_video_path)}")
                    os.remove(local_video_path)
                
                if final_gcs_path:
                    st.balloons()
                    st.header("🎉 Dubbing Complete!")
                    st.subheader("▶️ Play Your Dubbed Video")
                    signed_url = generate_download_signed_url_v4(gcs_client, config['BUCKET_NAME'], final_gcs_path)
                    if signed_url:
                        st.video(signed_url)
                    else:
                        st.warning("Could not generate a playable link for the video.")

                    authenticated_url = f"https://storage.cloud.google.com/{config['BUCKET_NAME']}/{final_gcs_path}"
                    st.success(f"Video saved to: [gs://{config['BUCKET_NAME']}/{final_gcs_path}]({authenticated_url})")
                else:
                    st.error("Processing failed. Please check the logs for details.")

if __name__ == "__main__":

    main()