import os
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
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- CONFIGURATION (Environment Variables) ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION", "us-central1")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
TTS_MODEL = os.environ.get("TTS_MODEL", "gemini-2.5-pro-tts")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-3-pro-preview")

# --- UI HELPER DATA (Same as app_local.py for consistency) ---
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

def download_blob(client, bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        return True
    except Exception as e:
        print(f"Failed to download file: {e}")
        return False

def upload_blob(client, bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        return f"gs://{bucket_name}/{destination_blob_name}"
    except Exception as e:
        print(f"Failed to upload file: {e}")
        return None

# --- CORE LOGIC (Copied from app_local.py) ---
def validate_and_fix_script(script, video_duration):
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
            print(f"⚠️ Skipping invalid segment timestamps: {seg}")
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
            print(f"⚠️ Overlap detected between segments ({overlap:.3f}s). Adjusting prev end time.")
            prev_seg['end_time'] = current_seg['start_time']
            
            if prev_seg['end_time'] <= prev_seg['start_time']:
                 fixed_script.pop()
        
        # Check for tiny gaps (< 100ms) - Merge
        elif 0 < (current_seg['start_time'] - prev_seg['end_time']) < 0.1:
             prev_seg['end_time'] = current_seg['start_time']
        
        fixed_script.append(current_seg)

    return fixed_script

def validate_and_refine_script(script, config):
    """
    Invokes Gemini again to validate the generated dubbing script for accuracy and timing.
    """
    print("🕵️ Validating and refining script with Gemini...")
    
    try:
        # Use Vertex AI for Cloud Run
        client = genai.Client(vertexai=True, project=config["PROJECT_ID"], location=config["LOCATION"])
            
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
        
        print("✅ Script validation complete.")
        return refined_script

    except Exception as e:
        print(f"⚠️ Script validation failed (using original script): {e}")
        return script

def get_dubbing_script_from_video(video_uri, config, video_duration):
    """
    Analyzes a video from GCS URI to produce a dubbing script.
    """
    try:
        print(f"Authenticating with Vertex AI (Project: {config['PROJECT_ID']}, Location: {config['LOCATION']})")
        client = genai.Client(vertexai=True, project=config["PROJECT_ID"], location=config["LOCATION"])
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return None

    print(f"🤖 Sending request to {config['MODEL_NAME']} for analysis...")
    
    video_part = types.Part.from_uri(
        file_uri=video_uri,
        mime_type="video/mp4" 
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
            print(f"🔄 Retry Attempt {current_attempt + 1}/{max_retries} due to validation error...")

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
                
            raw_script.sort(key=lambda x: float(x.get('end_time', 0)))
            last_timestamp = float(raw_script[-1]['end_time'])
            
            if last_timestamp > video_duration + 2.0:
                print(f"❌ Generated script duration ({last_timestamp:.2f}s) exceeds video duration ({video_duration:.2f}s).")
                base_prompt += f"\n\n🚨 **CRITICAL ERROR IN PREVIOUS ATTEMPT**: Your previous script ended at {last_timestamp}s, but the video is ONLY {video_duration}s long. You likely made a minute-to-second conversion error (e.g. 1:30 is 90s, not 130s). RE-CALCULATE TIMESTAMPS CAREFULLY."
                current_attempt += 1
                continue
            
            dubbing_script = raw_script
            break

        except Exception as e:
            print(f"❌ Gemini Analysis failed (Attempt {current_attempt + 1}): {e}")
            current_attempt += 1
            time.sleep(2)

    if not dubbing_script:
        print("❌ Failed to generate a valid script after multiple attempts.")
        return None

    try:
        print(f"📝 Raw Gemini Output: {json.dumps(dubbing_script, indent=2)}") 
        
        dubbing_script = validate_and_fix_script(dubbing_script, video_duration)
        dubbing_script = validate_and_refine_script(dubbing_script, config)
        dubbing_script = validate_and_fix_script(dubbing_script, video_duration)
        
        print("✅ Successfully received and parsed the dubbing script from Gemini.")
        return dubbing_script
    except (json.JSONDecodeError, IndexError, AttributeError) as e:
        print(f"❌ Failed to parse JSON from Gemini's response: {e}")
        return None

def extract_audio(video_path, audio_path):
    print(f"🎥 Extracting audio from '{os.path.basename(video_path)}'...")
    try:
        with VideoFileClip(video_path) as video_clip:
            video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le', logger=None)
        print(f"✅ Audio extracted to '{os.path.basename(audio_path)}'")
        return audio_path
    except Exception as e:
        print(f"❌ Error extracting audio: {e}")
        return None

def separate_background_music(audio_path, output_dir):
    print("🎶 Separating background music with Demucs... (This might take a while)")
    try:
        command = [
            sys.executable, "-m", "demucs.separate", "-n", "htdemucs",
            "-o", str(output_dir), "--two-stems", "vocals", str(audio_path)
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        
        audio_filename = os.path.splitext(os.path.basename(audio_path))[0]
        model_name = "htdemucs"
        background_path = os.path.join(output_dir, model_name, audio_filename, "no_vocals.wav")

        if os.path.exists(background_path):
            print("✅ Background music separated successfully.")
            return background_path
        else:
            print("❌ Demucs did not produce the background music file.")
            return None
    except Exception as e:
        print(f"❌ An error occurred during audio separation: {e}")
        return None

def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)

def synthesize_speech_with_gemini(text, segment_details, config):
    print(f"   Synthesizing: '{text[:40]}...' for {segment_details['speaker_label']}")
    try:
        client = genai.Client(vertexai=True, project=config["PROJECT_ID"], location=config["LOCATION"])
    except Exception as e:
        print(f"❌ TTS Authentication failed: {e}")
        return None

    audio_profile = f"Speaker: {segment_details['speaker_label']}, Gender: {segment_details['character_type']}, Language: {config['OUTPUT_LANGUAGE']}."
    scene_description = f"Movie dubbing session. The character is experiencing {segment_details['emotion']}. "
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
            print(f"❌ Gemini returned an empty or invalid response for the text. Skipping.")
            return None

        audio_data = response.candidates[0].content.parts[0].inline_data.data
        if not audio_data:
            print(f"❌ Gemini did not return audio data for the text.")
            return None

        wave_file(segment_details['output_path'], audio_data)
        return segment_details['output_path']
    except Exception as e:
        print(f"❌ Error synthesizing speech with Gemini: {e}")
        return None

def speed_up_audio(input_path, output_path, speed_factor):
    """Speeds up audio using ffmpeg's atempo filter."""
    try:
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

# --- FLASK ENDPOINTS ---

@app.route('/', methods=['POST'])
def process_video():
    """
    Expects JSON payload:
    {
        "gcs_uri": "gs://bucket/file.mp4",
        "input_language": "Chinese",
        "output_language": "Hindi"
    }
    """
    data = request.json
    gcs_uri = data.get('gcs_uri')
    input_lang = data.get('input_language', 'Chinese')
    output_lang = data.get('output_language', 'Hindi')

    if not gcs_uri or not BUCKET_NAME:
        return jsonify({"error": "Missing GCS URI or BUCKET_NAME env var"}), 400

    config = {
        "PROJECT_ID": PROJECT_ID,
        "LOCATION": LOCATION,
        "INPUT_LANGUAGE": input_lang,
        "OUTPUT_LANGUAGE": output_lang,
        "MODEL_NAME": LLM_MODEL,
        "TTS_MODEL": TTS_MODEL,
        "BUCKET_NAME": BUCKET_NAME
    }

    # Setup Workspace
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. Download Video
        local_video_path = os.path.join(temp_dir, "input_video.mp4")
        blob_name = gcs_uri.replace(f"gs://{BUCKET_NAME}/", "")
        print(f"Downloading {blob_name} to {local_video_path}")
        
        client = get_gcs_client()
        if not download_blob(client, BUCKET_NAME, blob_name, local_video_path):
             return jsonify({"error": "Failed to download video from GCS"}), 500
        
        config['local_video_path'] = local_video_path

        # 2. Extract & Separate Audio
        original_audio_path = os.path.join(temp_dir, "original.wav")
        extract_audio(local_video_path, original_audio_path)
        
        bg_dir = os.path.join(temp_dir, "separated")
        background_track_path = separate_background_music(original_audio_path, bg_dir)
        
        # 3. Analyze Video (Step 1)
        with VideoFileClip(local_video_path) as clip:
            video_duration = clip.duration
            
dubbing_script = get_dubbing_script_from_video(gcs_uri, config, video_duration)
        
        if not dubbing_script:
            return jsonify({"error": "Failed to generate dubbing script"}), 500

        # 4. Assign Voices (Auto-assign for Cloud Run version)
        # In a real API, you might return the script here and ask for assignments in a second call.
        # For now, we auto-assign using the logic.
        speaker_assignments = []
        speaker_info = {}
        for item in dubbing_script:
            speaker_info[item['speaker_label']] = item['character_type']
            
        voice_indices = {'MALE': 0, 'FEMALE': 0, 'CHILD': 0}
        sorted_speakers = sorted(speaker_info.keys())
        
        for speaker in sorted_speakers:
            char_type = speaker_info[speaker]
            if char_type == 'FEMALE':
                idx = voice_indices['FEMALE'] % len(FEMALE_VOICE_LIST)
                voice = FEMALE_VOICE_LIST[idx]
                voice_indices['FEMALE'] += 1
            elif char_type == 'MALE':
                idx = voice_indices['MALE'] % len(MALE_VOICE_LIST)
                voice = MALE_VOICE_LIST[idx]
                voice_indices['MALE'] += 1
            elif char_type == 'CHILD':
                idx = voice_indices['CHILD'] % len(CHILD_VOICE_LIST)
                voice = CHILD_VOICE_LIST[idx]
                voice_indices['CHILD'] += 1
            else:
                voice = FALLBACK_VOICE
            
            speaker_assignments.append({
                "speaker_label": speaker,
                "character_type": char_type,
                "selected_voice": voice
            })
            
        # 5. Synthesize (Step 2)
        # ... (Synthesis logic similar to app_local.py but adapted for linear execution)
        # For brevity, I will implement the core synthesis loop here directly
        
        final_video_clips = []
        final_audio_segments = []
        original_video = VideoFileClip(local_video_path)
        
        if background_track_path and os.path.exists(background_track_path):
             bg_audio_full = AudioSegment.from_wav(background_track_path)
        else:
             bg_audio_full = AudioSegment.silent(duration=int(original_video.duration * 1000))
             
        last_pos = 0.0
        
        for i, segment in enumerate(dubbing_script):
            output_text = segment.get(f"{output_lang}_translation", "...")
            start_time = segment['start_time']
            end_time = segment['end_time']
            start_time_ms = int(start_time * 1000)
            end_time_ms = int(end_time * 1000)
            
            # Gap
            if start_time > last_pos:
                gap_duration = start_time - last_pos
                if gap_duration > 0:
                    video_gap = original_video.subclipped(last_pos, start_time)
                    final_video_clips.append(video_gap)
                    gap_start_ms = int(last_pos * 1000)
                    gap_end_ms = int(start_time * 1000)
                    audio_gap = bg_audio_full[gap_start_ms:gap_end_ms]
                    final_audio_segments.append(audio_gap)
            
            # Speech
            selected_voice = next((item['selected_voice'] for item in speaker_assignments if item['speaker_label'] == segment['speaker_label']), FALLBACK_VOICE)
            orig_duration_ms = end_time_ms - start_time_ms
            
            segment_details = {
                'character_type': segment['character_type'],
                'emotion': segment.get('emotion', 'NEUTRAL'),
                'delivery_style': segment.get('delivery_style', 'NORMAL'),
                'speaker_label': segment.get('speaker_label', 'DEFAULT'),
                'pace': segment.get('pace', 'NORMAL'),
                'clip_duration': orig_duration_ms, 
                'selected_voice': selected_voice,
                'output_path': os.path.join(temp_dir, f"segment_{i}.wav")
            }
            
            synthesized_path = synthesize_speech_with_gemini(output_text, segment_details, config)
            
            if synthesized_path and os.path.exists(synthesized_path):
                with open(synthesized_path, "rb") as f:
                    dub_segment = AudioSegment.from_wav(f)
                
                dub_duration_ms = len(dub_segment)
                orig_duration_sec = orig_duration_ms / 1000.0
                dub_duration_sec = dub_duration_ms / 1000.0
                ratio = dub_duration_sec / orig_duration_sec if orig_duration_sec > 0 else 1.0
                
                video_clip = original_video.subclipped(start_time, end_time)
                bg_segment = bg_audio_full[start_time_ms:end_time_ms]
                
                if 0.9 <= ratio <= 1.1:
                    if ratio > 1.0:
                         speed_factor = ratio
                         fast_path = os.path.join(temp_dir, f"segment_{i}_fast.wav")
                         if speed_up_audio(synthesized_path, fast_path, speed_factor):
                             with open(fast_path, "rb") as f_fast:
                                 dub_segment = AudioSegment.from_wav(f_fast)
                    
                    mixed_segment = bg_segment.overlay(dub_segment)
                    final_audio_segments.append(mixed_segment)
                    final_video_clips.append(video_clip)

                elif ratio > 1.1:
                    # Audio longer -> Slow video
                    video_factor = 1.0 / ratio
                    video_clip = video_clip.with_speed_scaled(video_factor)
                    final_video_clips.append(video_clip)
                    
                    extended_bg = bg_segment * int(ratio + 2)
                    extended_bg = extended_bg[:dub_duration_ms]
                    mixed_segment = extended_bg.overlay(dub_segment)
                    final_audio_segments.append(mixed_segment)

                else: 
                    # Audio shorter -> Speed video
                    video_factor = 1.0 / ratio
                    video_clip = video_clip.with_speed_scaled(video_factor)
                    final_video_clips.append(video_clip)
                    
                    bg_segment = bg_segment[:dub_duration_ms]
                    mixed_segment = bg_segment.overlay(dub_segment)
                    final_audio_segments.append(mixed_segment)
            else:
                final_video_clips.append(original_video.subclipped(start_time, end_time))
                final_audio_segments.append(bg_audio_full[start_time_ms:end_time_ms])
            
            last_pos = end_time
            
        # End Gap
        if last_pos < original_video.duration:
             final_video_clips.append(original_video.subclipped(last_pos, original_video.duration))
             final_audio_segments.append(bg_audio_full[int(last_pos*1000):])
             
        # Concatenate
        final_audio_track = sum(final_audio_segments, AudioSegment.empty())
        final_audio_path = os.path.join(temp_dir, "final_audio.wav")
        final_audio_track.export(final_audio_path, format="wav")
        
        final_video_clip = concatenate_videoclips(final_video_clips)
        final_video_path = os.path.join(temp_dir, "output_video.mp4")
        
        final_video_clip.write_videofile(
            final_video_path, 
            codec="libx264", 
            audio=final_audio_path, 
            audio_codec="aac", 
            logger=None
        )
        
        # Upload Result
        output_blob = f"outputs/{os.path.basename(gcs_uri)}_{output_lang}.mp4"
        result_uri = upload_blob(client, BUCKET_NAME, final_video_path, output_blob)
        
        return jsonify({"status": "success", "output_uri": result_uri})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))