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
MALE_VOICE_LIST = ['Puck', 'Orus', 'Enceladus']
FEMALE_VOICE_LIST = ['Kore', 'Zephyr', 'Leda', 'Sulafat', 'Aoede']
CHILD_VOICE_LIST = ['Leda', 'Kore']
FALLBACK_VOICE = 'Leda'

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
    """Lists directories and files in a GCS path."""
    if prefix and not prefix.endswith('/'):
        prefix += '/'
    
    blobs = client.list_blobs(bucket_name, prefix=prefix, delimiter='/')
    
    dirs = []
    files = []
    
    # This is a bit of magic to get the 'subdirectories'
    if hasattr(blobs, 'prefixes') and blobs.prefixes:
        dirs = [p.rstrip('/') for p in blobs.prefixes]

    video_extensions = ['.mp4', '.mov', '.avi', '.mkv']
    for blob in blobs:
        if any(blob.name.lower().endswith(ext) for ext in video_extensions):
            files.append(blob.name)
            
    return dirs, files

def download_gcs_file(client, bucket_name, source_blob_name, logger):
    """Downloads a blob from the bucket to a local temporary file."""
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        
        # Create a temporary file and get its path
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(source_blob_name)[1]) as tmpfile:
            logger.log(f"⬇️ Downloading gs://{bucket_name}/{source_blob_name} to a temporary file...")
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

        # This URL is valid for 1 hour
        url = blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(hours=1),
            method="GET",
        )
        return url
    except Exception as e:
        st.error(f"Could not generate download link. Error: {e}")
        return None
    
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

def get_dubbing_script_from_video(video_path, config, logger):
    """
    Uploads a video to the Gemini API, analyzes it to produce a
    timestamped, translated, and character-identified dubbing script.
    """
    try:
        if config["USE_VERTEX_AI"]:
            logger.log(f"Authenticating with Vertex AI (Project: {config['PROJECT_ID']}, Location: {config['LOCATION']})")
            client = genai.Client(project=config["PROJECT_ID"], location=config["LOCATION"])
        else:
            logger.log("Authenticating with Google API Key")
            #genai.configure(api_key=config["GOOGLE_API_KEY"])
            client = genai.Client(api_key=config["GOOGLE_API_KEY"])
    except Exception as e:
        logger.log(f"❌ Authentication failed: {e}")
        return None

    logger.log(f"Uploading video '{os.path.basename(video_path)}' to the Gemini API...")
    video_file = client.files.upload(file=video_path)

    processing_message = "...Video is processing"
    logger.log(processing_message)
    while video_file.state.name == "PROCESSING":
        time.sleep(10)
        video_file = client.files.get(name=video_file.name)

    if video_file.state.name == "FAILED":
        logger.log(f"❌ Video processing failed: {video_file.state}")
        raise ValueError(f"Video processing failed: {video_file.state}")

    logger.log(f"✅ Video uploaded and processed successfully: {video_file.name}")

    prompt = f"""
    You are an expert voice director and video producer creating a script for dubbing.
    Analyze the provided video file's audio track with extreme detail. Your goal is to capture the complete performance, including the emotional and dramatic context.

    Follow these steps precisely:
    1.  **Speaker Diarization**: Use a combination of audio and video to identify every distinct speaker and assign a unique label (e.g., SPEAKER_1).
    2.  **Character Classification**: Classify each speaker as MALE, FEMALE, or CHILD.
    3.  **Emotional Analysis**: For each dialogue segment, identify the primary emotion being conveyed. Choose from this list: **HAPPY, SAD, ANGRY, SURPRISED, FEARFUL, NEUTRAL**.
    4.  **Delivery Style Analysis**: For each dialogue segment, identify the style of delivery. Choose from this list: **NORMAL, SHOUTING, WHISPERING**.
    5.  **Transcription & Translation**: Provide the timestamped original '{config["INPUT_LANGUAGE"]}' transcript and its accurate and meaningful '{config["OUTPUT_LANGUAGE"]}' translation as used in Hindi movies.
    6.  **Pace of Speech: For each dialogue segment, identify the pace of delivery. Choose from this list: **NORMAL, FAST, VERY FAST, SLOW, MEDIUM, VERY SLOW**.
    7.  **Always assign the start and end time in seconds. Do not consider minutes. Eg. If time shows 1.12 then it should be 60+12=72 seconds and not 112 seconds**.
    8.  **Output Format**: Your final output MUST be a valid JSON array of objects. Do not include any text or explanations outside of this array. Each object represents a single line of dialogue and must have the following structure:
        {{
          "start_time": float,
          "end_time": float,
          "speaker_label": "string",
          "character_type": "string (MALE, FEMALE, or CHILD)",
          "emotion": "string (e.g., ANGRY, HAPPY, NEUTRAL)",
          "delivery_style": "string (NORMAL, SHOUTING, or WHISPERING)",
          "original_transcript": "string ('{config["INPUT_LANGUAGE"]}' text)",
          "{config["OUTPUT_LANGUAGE"]}_translation": "string ({config["OUTPUT_LANGUAGE"]} text)",
          "pace": "string (NORMAL, FAST, VERY FAST, SLOW, MEDIUM or VERY SLOW)"
        }}
    """

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
        with st.expander("Show Generated Dubbing Script (JSON)"):
            st.json(dubbing_script)
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
    logger.log("🎶 Separating background music with Demucs... (This might take a while)")
    try:
        command = [
            "python3", "-m", "demucs.separate", "-n", "htdemucs",
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
            client = genai.Client(project=config["PROJECT_ID"], location=config["LOCATION"])
        else:
            #genai.configure(api_key=config["GOOGLE_API_KEY"])
            client = genai.Client(api_key=config["GOOGLE_API_KEY"])
    except Exception as e:
        logger.log(f"❌ TTS Authentication failed: {e}")
        return None

    voice_description = f"a standard '{config['OUTPUT_LANGUAGE']}' voice"
    if segment_details['character_type'] == "MALE":
        voice_description = f"a standard '{config['OUTPUT_LANGUAGE']}' MALE voice"
    elif segment_details['character_type'] == "FEMALE":
        voice_description = f"a standard '{config['OUTPUT_LANGUAGE']}' FEMALE voice"
    elif segment_details['character_type'] == "CHILD":
        voice_description = f"a higher-pitched '{config['OUTPUT_LANGUAGE']}' child's voice"

    style_description = ""
    if segment_details['delivery_style'] == "SHOUTING":
        style_description = " in a loud, shouting tone"
    elif segment_details['delivery_style'] == "WHISPERING":
        style_description = " in a soft, whispering tone"
    elif segment_details['delivery_style'] == "NORMAL":
        style_description = "in a normal tone"
    elif segment_details['emotion'] not in ["NEUTRAL", "NORMAL"]:
        style_description = f" in a {segment_details['emotion'].lower()} tone"

    instruction = f"You are a movie dubbing expert and need to deliver audio as professional editor. Maintain the duration of the output audio within {segment_details['clip_duration']} milliseconds."
    #instruction = f"You are a movie dubbing expert. Maintain the duration of the output audio within {segment_details['clip_duration']} milliseconds."
    #full_prompt = f"{instruction}. Using {voice_description} {style_description} with {segment_details['emotion']} emotion, {segment_details['speaker_label']} says the following at {pace} speed: {text}"

    full_prompt = f"{instruction}. Using {voice_description} {style_description} with {segment_details['emotion']} emotion, {segment_details['speaker_label']} says the following at {segment_details['pace']} speed: {text}"

    try:
        #model_to_use = client.models.get_model(config['TTS_MODEL'])
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

def assign_specific_voices(transcript_data):
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
            "speaker_label": speaker, "character_type": char_type, "selected_voice": selected_voice
        })
    return speaker_voice_array

def process_video_dubbing(video_path, config, logger):
    """Main function to orchestrate the entire dubbing process."""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, f"output_{base_name}")
        os.makedirs(output_dir, exist_ok=True)
        
        original_audio_path = os.path.join(output_dir, f"{base_name}_original_audio.wav")
        extract_audio(video_path, original_audio_path, logger)
        
        background_track_path = separate_background_music(original_audio_path, os.path.join(output_dir, "separated"), logger)
        
        dubbing_script = get_dubbing_script_from_video(video_path, config, logger)
        if not dubbing_script:
            logger.log("❌ Aborting due to failure in script generation.")
            return None

        logger.log(f"🔊 Initializing {config['TTS_MODEL']} for Text-to-Speech...")
        logger.log("🎤 Starting speech synthesis with Gemini...")

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
        total_segments = len(dubbing_script)
        progress_bar = st.progress(0, text="Synthesizing audio segments...")

        for i, segment in enumerate(dubbing_script):
            output_text = segment.get(output_lang_key, "...")
            start_time_ms = int(segment['start_time'] * 1000)
            end_time_ms = int(segment['end_time'] * 1000)
            
            selected_voice = next((item['selected_voice'] for item in speaker_assignments if item['speaker_label'] == segment['speaker_label']), FALLBACK_VOICE)
            
            segment_details = {
                'character_type': segment['character_type'],
                'emotion': segment.get('emotion', 'NEUTRAL'),
                'delivery_style': segment.get('delivery_style', 'NORMAL'),
                'speaker_label': segment.get('speaker_label', 'DEFAULT'),
                'pace': segment.get('pace', 'NORMAL'),
                'clip_duration': end_time_ms - start_time_ms,
                'selected_voice': selected_voice,
                'output_path': os.path.join(output_dir, f"segment_{i}.wav")
            }
            
            time.sleep(2) 
            
            synthesized_path = synthesize_speech_with_gemini(output_text, segment_details, config, logger)
            
            if synthesized_path and os.path.exists(synthesized_path):
                with open(synthesized_path, "rb") as f:
                    dub_segment = AudioSegment.from_wav(f)
                final_vocal_track = final_vocal_track.overlay(dub_segment, position=start_time_ms)
                os.remove(synthesized_path)
            else:
                logger.log(f"⚠️ WARNING: Segment {i} could not be synthesized and will be silent.")
            
            progress_bar.progress((i + 1) / total_segments, text=f"Synthesizing audio segment {i+1}/{total_segments}")
            
        logger.log("🎤 Speech synthesis complete. Combining audio tracks...")
        final_audio_track = background_music.overlay(final_vocal_track)
        final_audio_path = os.path.join(output_dir, f"{base_name}_dubbed_audio.wav")
        final_audio_track.export(final_audio_path, format="wav")
        logger.log("✅ Final audio track created.")

        # <<< --- MODIFIED LOGIC --- >>>
        final_video_path = os.path.join(output_dir, f"dubbed_{base_name}.mp4")
        merged_video_path = merge_audio_with_video(video_path, final_audio_path, final_video_path, logger)
        
        # Upload the final video to GCS
        if merged_video_path and os.path.exists(merged_video_path):
            gcs_client = get_gcs_client()
            destination_blob_name = f"dubbed_videos/{os.path.basename(merged_video_path)}"
            
            upload_successful = upload_to_gcs(
                gcs_client, 
                config['BUCKET_NAME'], 
                merged_video_path, 
                destination_blob_name, 
                logger
            )
            
            if upload_successful:
                # Return the GCS path of the uploaded file
                return destination_blob_name 
        
        return None # Return None if anything failed

# --- STREAMLIT UI ---
def main():
    st.set_page_config(layout="wide", page_title="Gemini Video Dubber")
    st.title("🎬 Gemini Video Dubbing Studio")
    st.markdown("Select a video from Google Cloud Storage, choose your languages, and let Gemini automatically dub it for you.")

    gcs_client = get_gcs_client()

    if 'current_path' not in st.session_state:
        st.session_state.current_path = ""
    if 'selected_bucket' not in st.session_state:
        st.session_state.selected_bucket = None

    with st.sidebar:
        st.header("🛠️ Configuration")

        with st.form("config_form"):
            google_api_key = st.text_input("Google API Key", type="password", help="Required if not using Vertex AI.")
            use_vertex_ai = st.checkbox("Use Vertex AI", value=False)
            
            project_id = None
            location = None
            if use_vertex_ai:
                project_id = st.text_input("Google Cloud Project ID", help="Required for Vertex AI.")
                location = st.selectbox("GCP Location", options=GCP_REGIONS, index=0, help="The region for your Vertex AI resources.")

            st.subheader("📁 Select Video from GCS")
            
            buckets = list_gcs_buckets(gcs_client)
            selected_bucket = st.selectbox("1. Select GCS Bucket", options=[""] + buckets, key="selected_bucket")
            
            gcs_file_path = None
            if selected_bucket:
                if 'prev_bucket' not in st.session_state or st.session_state.prev_bucket != selected_bucket:
                    st.session_state.current_path = ""
                st.session_state.prev_bucket = selected_bucket

                dirs, files = list_gcs_dirs_files(gcs_client, selected_bucket, st.session_state.current_path)
                
                current_display_path = f"{selected_bucket}/{st.session_state.current_path}"
                st.info(f"Current Path: `{current_display_path}`")

                nav_options = []
                if st.session_state.current_path:
                    nav_options.append("⬆️ Go Up (..)")
                nav_options.extend([f"📁 {d.split('/')[-1]}" for d in dirs])
                
                nav_choice = st.selectbox("2. Navigate Folders", options=[""] + nav_options)
                
                # This block now updates the state and lets Streamlit's natural rerun handle the UI update
                if nav_choice:
                    if nav_choice == "⬆️ Go Up (..)":
                        st.session_state.current_path = '/'.join(st.session_state.current_path.rstrip('/').split('/')[:-1])
                    else:
                        folder_name = nav_choice.replace("📁 ", "")
                        st.session_state.current_path = os.path.join(st.session_state.current_path, folder_name)
                    # The st.experimental_rerun() line was removed from here
                
                file_options = [f.split('/')[-1] for f in files]
                selected_file = st.selectbox("3. Select Video File", options=[""] + file_options)
                
                if selected_file:
                    gcs_file_path = os.path.join(st.session_state.current_path, selected_file)

            st.subheader("⚙️ Dubbing Settings")
            col1, col2 = st.columns(2)
            with col1:
                input_lang = st.selectbox("Input Language", options=LANGUAGES, index=LANGUAGES.index("Chinese"))
            with col2:
                output_lang = st.selectbox("Output Language", options=LANGUAGES, index=LANGUAGES.index("Hindi"))

            llm_model_name = st.selectbox("LLM Model Name", options=["gemini-2.5-pro", "gemini-2.5-flash"], index=0)
            tts_model_name = st.selectbox("TTS Model Name (Preview)", options=["gemini-2.5-pro-preview-tts", "gemini-2.5-flash-preview-tts"], index=0)
            
            submitted = st.form_submit_button("Start Dubbing Process")

    if submitted:
        if not use_vertex_ai and not google_api_key:
            st.error("Please provide a Google API Key or select 'Use Vertex AI'.")
        elif use_vertex_ai and (not project_id or not location):
            st.error("Please provide a Project ID and Location for Vertex AI.")
        elif not gcs_file_path:
            st.error("Please select a video file from Google Cloud Storage.")
        else:
            logger = StatusLogger()
            
            local_video_path = download_gcs_file(gcs_client, selected_bucket, gcs_file_path, logger)

            if local_video_path:
                config = {
                    "GOOGLE_API_KEY": google_api_key,
                    "USE_VERTEX_AI": use_vertex_ai,
                    "PROJECT_ID": project_id,
                    "LOCATION": location,
                    "INPUT_LANGUAGE": input_lang,
                    "OUTPUT_LANGUAGE": output_lang,
                    "MODEL_NAME": f"models/{llm_model_name}",
                    "TTS_MODEL": tts_model_name,
                    "BUCKET_NAME": selected_bucket,
                }
                
                st.info(f"Starting dubbing process for **gs://{selected_bucket}/{gcs_file_path}**...")
                
                final_gcs_path = None
                try:
                    final_gcs_path = process_video_dubbing(local_video_path, config, logger)
                except Exception as e:
                    logger.log(f"❌ An unexpected error occurred in the main process: {e}")
                    st.exception(e)
                finally:
                    logger.log(f"🧹 Cleaning up temporary file: {os.path.basename(local_video_path)}")
                    os.remove(local_video_path)
                
                # <<< --- MODIFIED LOGIC IS HERE --- >>>
                if final_gcs_path:
                    st.balloons()
                    st.header("🎉 Dubbing Complete!")
                    
                    # Construct the GCS Authenticated URL
                    authenticated_url = f"https://storage.cloud.google.com/{selected_bucket}/{final_gcs_path}"
                    
                    st.success("Your dubbed video has been saved to GCS!")
                    st.markdown(f"You can access it here: **[GCS Authenticated Link]({authenticated_url})**")
                    st.info("Note: You must be logged into a Google account with permission to view this bucket to access the link.")
                else:
                    st.error("Processing failed. No output video was generated. Please check the logs above for details.")
                    
if __name__ == "__main__":
    main()