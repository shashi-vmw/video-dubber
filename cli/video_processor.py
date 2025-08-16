import os
import time
import shutil
from moviepy import VideoFileClip
from pydub import AudioSegment

from file_manager import FileManager
from video_splitter import VideoSplitter
from dubbing_script_generator import DubbingScriptGenerator
from audio_processor import AudioProcessor


class VideoProcessor:
    """Main orchestrator for the video dubbing pipeline."""
    
    def __init__(self, working_dir="working-dir"):
        self.file_manager = FileManager(working_dir)
        self.video_splitter = VideoSplitter()
        self.script_generator = DubbingScriptGenerator(self.file_manager)
        self.audio_processor = AudioProcessor()
    
    def process_video_dubbing(self, video_path, output_path, config, logger):
        """Main function to orchestrate the entire dubbing process with automatic splitting for large files."""
        overall_start_time = time.time()
        
        self._log_pipeline_start(video_path, output_path, config, logger)
        
        try:
            reuse_path = config.get("REUSE_PATH")
            
            # Set up working directory
            run_dir = self.file_manager.setup_working_directory(video_path, reuse_path=reuse_path)
            logger(f"📂 Working directory: {run_dir}")
            
            # Create output directory
            output_dir = self.file_manager.setup_output_directory(output_path)
            logger(f"📁 Output directory: {output_dir}")
            
            # Step 1: Handle video source (compression or reuse)
            if reuse_path:
                processing_video = self._find_video_in_reuse_dir(run_dir, logger)
                if not processing_video:
                    return None
                logger(f"✅ Reusing video from previous run: {os.path.basename(processing_video)}")
            else:
                processing_video = self._handle_compression(video_path, run_dir, config, logger)
            
            # Step 2: Check if video needs splitting
            video_segments = self._handle_video_splitting(processing_video, run_dir, logger)
            if not video_segments:
                return None
            
            # Step 3: Process each segment sequentially
            processed_segments = self._process_segments(video_segments, config, logger)
            if not processed_segments:
                return None

            # If in extraction-only mode, we're done.
            if config.get("EXTRACTION_ONLY"):
                logger(f"\n{'='*80}")
                logger("✅ EXTRACTION-ONLY MODE COMPLETED")
                logger(f"{'='*80}")
                logger(f"Artifacts have been extracted to the working directory:")
                logger(f"📁 {run_dir}")
                return run_dir

            # Step 4: Combine segments if needed and place in output directory
            final_output = self._handle_segment_combination(processed_segments, output_dir, video_path, logger)
            if not final_output:
                return None
            
            # Step 5: Cleanup (skip for reuse mode)
            if not reuse_path:
                self._handle_cleanup(video_segments, processing_video, video_path, logger)
            else:
                logger("\n🧹 Skipping cleanup in reuse mode.")

            # Final summary
            self._log_pipeline_completion(overall_start_time, video_path, output_dir, config, final_output, logger)
            
            return final_output
            
        except Exception as e:
            self._log_pipeline_failure(overall_start_time, video_path, e, logger)
            return None

    def process_single_segment(self, segment_path, config, logger, segment_index=1, total_segments=1):
        """Process a single video segment through the dubbing pipeline."""
        logger(f"🎬 Processing segment {segment_index}/{total_segments}: {os.path.basename(segment_path)}")
        
        base_name = os.path.splitext(os.path.basename(segment_path))[0]
        output_dir = self.file_manager.get_segment_dir(segment_index, base_name)
        final_audio_path = os.path.join(output_dir, f"{base_name}_dubbed_audio.wav")

        # If in reuse mode and the FINAL audio already exists, we can skip generation entirely.
        if config.get("REUSE_PATH") and os.path.exists(final_audio_path):
            logger(f"✅ Final dubbed audio already exists. Reusing: {os.path.basename(final_audio_path)}")
        else:
            # Otherwise, run the generation process, which has its own internal reuse logic.
            logger("🎧 Dubbed audio not found or reuse is disabled. Starting generation process...")
            final_audio_path = self._generate_dubbed_audio(
                segment_path, config, logger, segment_index, base_name, output_dir
            )
            if not final_audio_path:
                logger(f"❌ Audio generation failed for segment {segment_index}.")
                return None

        # Merge the final audio with the video segment.
        segment_output = os.path.join(output_dir, f"dubbed_{base_name}.mp4")
        merged_video_path = self.audio_processor.merge_audio_with_video(
            segment_path, final_audio_path, segment_output, logger
        )
        
        return merged_video_path

    def _extract_artifacts_for_segment(self, segment_path, config, logger, segment_index, total_segments):
        """Runs only the extraction steps for a single video segment."""
        logger(f"🔍 Extracting artifacts for segment {segment_index}/{total_segments}: {os.path.basename(segment_path)}")
        base_name = os.path.splitext(os.path.basename(segment_path))[0]
        output_dir = self.file_manager.get_segment_dir(segment_index, base_name)

        # Step 1: Extract Original Audio
        original_audio_path = os.path.join(output_dir, f"{base_name}_original_audio.wav")
        if not self.audio_processor.extract_audio(segment_path, original_audio_path, logger):
            return None

        # Step 2: Separate Background Music
        separated_dir = os.path.join(output_dir, "separated")
        self.audio_processor.separate_background_music(original_audio_path, separated_dir, logger)
        
        logger(f"✅ Artifact extraction complete for segment {segment_index}.")
        return output_dir # Return a success indicator

    def _generate_dubbed_audio(self, segment_path, config, logger, segment_index, base_name, output_dir):
        """
        Generate the final dubbed audio track, intelligently reusing and regenerating 
        artifacts as needed.
        """
        # Step 1: Ensure Original Audio Exists
        original_audio_path = os.path.join(output_dir, f"{base_name}_original_audio.wav")
        if not os.path.exists(original_audio_path):
            logger("🎤 Original audio not found, extracting from video...")
            if not self.audio_processor.extract_audio(segment_path, original_audio_path, logger):
                return None # Critical error, cannot proceed
        else:
            logger("✅ Reusing existing original audio.")

        # Step 2: Ensure Background Music Exists
        separated_dir = os.path.join(output_dir, "separated")
        background_track_path = os.path.join(separated_dir, "htdemucs", base_name, "no_vocals.wav")
        if not os.path.exists(background_track_path):
            logger("🎶 Background music not found, running separation...")
            background_track_path = self.audio_processor.separate_background_music(
                original_audio_path, separated_dir, logger
            )
            if not background_track_path:
                logger("⚠️ Music separation failed. Proceeding without background music.")
        else:
            logger("✅ Reusing existing background music.")

        # Step 3: Generate Dubbing Script (with reuse)
        dubbing_script = self.script_generator.generate_dubbing_script(segment_path, config, logger)
        if not dubbing_script:
            return None

        # Step 4: Prepare Audio Assets & Synthesize Vocals
        video_duration_ms = self._get_video_duration_ms(segment_path)
        background_music = self._prepare_background_music(background_track_path, video_duration_ms, logger)
        speaker_assignments = self.script_generator.assign_voices_to_speakers(dubbing_script)
        
        final_vocal_track = self.audio_processor.process_audio_segments(
            dubbing_script, speaker_assignments, config, output_dir, background_music, logger
        )
        if final_vocal_track is None:
            return None

        # Step 5: Mix and Export Final Audio
        logger("🎵 Combining background music with dubbed vocals...")
        if final_vocal_track.max > 0:
            final_audio_track = background_music.overlay(final_vocal_track)
            logger("✅ Audio tracks combined successfully.")
        else:
            logger("⚠️ No vocal content found, using background music only.")
            final_audio_track = background_music
        
        final_audio_path = os.path.join(output_dir, f"{base_name}_dubbed_audio.wav")
        final_audio_track.export(final_audio_path, format="wav")
        logger(f"💾 Final audio saved: {os.path.basename(final_audio_path)}")
        
        return final_audio_path
    
    def _find_video_in_reuse_dir(self, run_dir, logger):
        """Find the video file in a reuse directory."""
        supported_formats = ['.mp4', '.mov', '.avi', '.mkv']
        for item in os.listdir(run_dir):
            if os.path.isfile(os.path.join(run_dir, item)) and any(item.lower().endswith(fmt) for fmt in supported_formats):
                # Avoid picking up dubbed segment files
                if 'dubbed_' not in item and 'segment_' not in item:
                    return os.path.join(run_dir, item)
        logger(f"❌ Could not find a suitable video file to reuse in '{run_dir}'")
        return None

    def _log_pipeline_start(self, video_path, output_path, config, logger):
        """Log the start of the pipeline with configuration details."""
        logger(f"\n{'='*80}")
        logger(f"🚀 STARTING VIDEO DUBBING PIPELINE")
        logger(f"{'='*80}")
        logger(f"📁 Input: {os.path.basename(video_path)}")
        logger(f"📁 Output: {os.path.basename(output_path)}")
        logger(f"🌍 Languages: {config.get('INPUT_LANGUAGE', 'Unknown')} → {config.get('OUTPUT_LANGUAGE', 'Unknown')}")
        logger(f"🤖 Model: {config.get('MODEL_NAME', 'Unknown')}")
        logger(f"🎤 TTS Model: {config.get('TTS_MODEL', 'Unknown')}")
        
        profile = config.get("COMPRESSION_PROFILE")
        logger(f"🗜️ Compression: {profile if profile else 'Disabled'}")

        if config.get("REUSE_PATH"):
            logger(f"🔄 Reuse Mode: Enabled (Path: {config['REUSE_PATH']})")
        if config.get("STRICT", False):
            logger("🛡️ Strict Mode: Enabled")
        if config.get("EXTRACTION_ONLY", False):
            logger("🔍 Extraction-Only Mode: Enabled")
    
    def _handle_compression(self, video_path, run_dir, config, logger):
        """Handle video compression if requested."""
        profile = config.get("COMPRESSION_PROFILE")
        if profile:
            logger(f"\n{'='*60}")
            logger("🗜️ STEP 1: COMPRESSION")
            logger(f"{'='*60}")
            
            compression_start = time.time()
            compressed_video = self.video_splitter.compress_video_for_testing(video_path, run_dir, profile, logger)
            compression_end = time.time()
            
            if compressed_video:
                logger(f"✅ Compression completed in {compression_end - compression_start:.1f} seconds")
                logger(f"   📁 Using compressed video: {os.path.basename(compressed_video)}")
                return compressed_video
            else:
                logger("❌ Compression failed. Aborting pipeline.")
                return None
        else:
            logger(f"\n{'='*60}")
            logger("📹 STEP 1: USING ORIGINAL VIDEO (No compression)")
            logger(f"{'='*60}")
            logger(f"   📁 Processing: {os.path.basename(video_path)}")
            return video_path
    
    def _handle_video_splitting(self, processing_video, run_dir, logger):
        """Handle video splitting if needed."""
        logger(f"\n{'='*60}")
        logger("✂️ STEP 2: VIDEO ANALYSIS & SPLITTING")
        logger(f"{'='*60}")
        
        splitting_start = time.time()
        video_segments = self.video_splitter.split_video_by_size(processing_video, run_dir, logger)
        splitting_end = time.time()
        
        if not video_segments:
            logger("❌ Failed to analyze/split video")
            return None
            
        logger(f"✅ Video analysis completed in {splitting_end - splitting_start:.1f} seconds")
        logger(f"   📊 Result: {len(video_segments)} segment(s) to process")
        
        return video_segments
    
    def _process_segments(self, video_segments, config, logger):
        """Process all video segments sequentially."""
        step_name = "EXTRACTING ARTIFACTS" if config.get("EXTRACTION_ONLY") else "DUBBING SEGMENTS"
        logger(f"\n{'='*60}")
        logger(f"🎬 STEP 3: {step_name} ({len(video_segments)} segment(s))")
        logger(f"{'='*60}")
        
        processed_segments = []
        total_segments = len(video_segments)
        segment_times = []
        
        for i, segment_path in enumerate(video_segments, 1):
            segment_start_time = time.time()
            logger(f"\n🎯 Processing Segment {i}/{total_segments}")
            logger(f"   📁 Input: {os.path.basename(segment_path)}")
            logger(f"   📊 Size: {self.file_manager.get_file_size_gb(segment_path):.2f} GB")
            
            if config.get("EXTRACTION_ONLY"):
                result = self._extract_artifacts_for_segment(segment_path, config, logger, i, total_segments)
            else:
                result = self.process_single_segment(segment_path, config, logger, i, total_segments)

            segment_end_time = time.time()
            segment_duration = segment_end_time - segment_start_time
            segment_times.append(segment_duration)
            
            if not result:
                logger(f"❌ Failed to process segment {i}/{total_segments}")
                return None
                
            processed_segments.append(result)
            logger(f"✅ Segment {i}/{total_segments} completed in {segment_duration/60:.1f} minutes")
            
            # Show progress and estimated time remaining
            if i < total_segments:
                self._log_segment_progress(segment_times, i, total_segments, logger)
        
        self._log_segments_completion(segment_times, logger)
        return processed_segments
    
    def _handle_segment_combination(self, dubbed_segments, output_dir, video_path, logger):
        """Handle combining segments or finalizing single segment."""
        # Create final output filename based on input video
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        final_output_path = os.path.join(output_dir, f"{base_name}_dubbed.mp4")
        
        if len(dubbed_segments) > 1:
            logger(f"\n{'='*60}")
            logger("🔗 STEP 4: COMBINING SEGMENTS")
            logger(f"{'='*60}")
            
            combination_start = time.time()
            final_output = self.video_splitter.combine_dubbed_segments(dubbed_segments, final_output_path, logger)
            combination_end = time.time()
            
            if final_output:
                logger(f"✅ Segments combined in {combination_end - combination_start:.1f} seconds")
                return final_output
            else:
                logger(f"❌ Failed to combine segments")
                return None
        else:
            return self._finalize_single_segment(dubbed_segments[0], final_output_path, logger)
    
    def _finalize_single_segment(self, dubbed_segment, output_path, logger):
        """Finalize processing for a single segment."""
        logger(f"\n{'='*60}")
        logger("📁 STEP 4: FINALIZING SINGLE SEGMENT")
        logger(f"{'='*60}")
        
        try:
            if os.path.exists(dubbed_segment):
                shutil.copy2(dubbed_segment, output_path)
                logger(f"✅ Single segment copied to final location")
                logger(f"   📁 Source: {dubbed_segment}")
                logger(f"   📁 Final: {output_path}")
                return output_path
            else:
                logger(f"❌ Dubbed segment file not found: {dubbed_segment}")
                return None
        except Exception as e:
            logger(f"❌ Failed to copy final file: {e}")
            logger(f"   💬 Source: {dubbed_segment}")
            logger(f"   💬 Destination: {output_path}")
            logger(f"   💬 Error type: {type(e).__name__}")
            return None
    
    def _handle_cleanup(self, video_segments, processing_video, video_path, logger):
        """Handle cleanup of temporary files."""
        logger(f"\n{'='*60}")
        logger("🧹 STEP 5: CLEANUP")
        logger(f"{'='*60}")
        
        # Clean up split segments if they were created
        temp_files = []
        if len(video_segments) > 1 and video_segments[0] != video_path:
            temp_files.extend(video_segments)
        
        self.file_manager.cleanup_temp_files(temp_files, logger)
    
    def _log_pipeline_completion(self, start_time, video_path, output_path, config, final_output, logger):
        """Log successful pipeline completion."""
        total_duration = time.time() - start_time
        
        logger(f"\n{'='*80}")
        logger("🎉 DUBBING PIPELINE COMPLETED SUCCESSFULLY!")
        logger(f"{'='*80}")
        logger(f"📊 Final Summary:")
        logger(f"   • Input: {os.path.basename(video_path)}")
        logger(f"   • Output: {os.path.basename(output_path)}")
        logger(f"   • Languages: {config.get('INPUT_LANGUAGE')} → {config.get('OUTPUT_LANGUAGE')}")
        logger(f"   • Total time: {total_duration/60:.1f} minutes")
        logger(f"   • Output size: {self.file_manager.get_file_size_gb(final_output):.2f} GB")
        logger(f"📁 Final video: {output_path}")
    
    def _log_pipeline_failure(self, start_time, video_path, error, logger):
        """Log pipeline failure with debugging information."""
        total_duration = time.time() - start_time
        
        logger(f"\n{'='*80}")
        logger("❌ DUBBING PIPELINE FAILED")
        logger(f"{'='*80}")
        logger(f"🐛 Error: {str(error)}")
        logger(f"🐛 Error type: {type(error).__name__}")
        logger(f"⏱️ Time elapsed before failure: {total_duration/60:.1f} minutes")
        logger(f"📁 Input file: {video_path}")
        
        # Provide helpful debugging tips
        error_str = str(error).lower()
        if "ffmpeg" in error_str:
            logger(f"💡 Tip: This appears to be an FFmpeg error. Ensure FFmpeg is installed and accessible.")
        elif "gemini" in error_str or "api" in error_str:
            logger(f"💡 Tip: This appears to be an API error. Check your API key and network connection.")
        elif "file" in error_str or "path" in error_str:
            logger(f"💡 Tip: This appears to be a file system error. Check file paths and permissions.")
    
    def _log_segment_progress(self, segment_times, current_segment, total_segments, logger):
        """Log progress and estimated time remaining."""
        avg_time = sum(segment_times) / len(segment_times)
        remaining_segments = total_segments - current_segment
        estimated_remaining = (avg_time * remaining_segments) / 60
        logger(f"   ⏱️ Progress: {current_segment}/{total_segments} ({current_segment/total_segments*100:.1f}%)")
        logger(f"   🔮 Estimated time remaining: {estimated_remaining:.1f} minutes")
    
    def _log_segments_completion(self, segment_times, logger):
        """Log completion of all segments."""
        total_processing_time = sum(segment_times)
        logger(f"\n✅ All segments processed successfully!")
        logger(f"   ⏱️ Total dubbing time: {total_processing_time/60:.1f} minutes")
        logger(f"   📊 Average per segment: {(total_processing_time/len(segment_times))/60:.1f} minutes")
    
    def _get_video_duration_ms(self, video_path):
        """Get video duration in milliseconds."""
        with VideoFileClip(video_path) as clip:
            return int(clip.duration * 1000)
    
    def _prepare_background_music(self, background_track_path, video_duration_ms, logger):
        """Prepare background music track with fallback options."""
        if background_track_path and os.path.exists(background_track_path):
            try:
                background_music = AudioSegment.from_wav(background_track_path)
                
                # Check if background music has meaningful content (not just silence)
                if background_music.max_possible_amplitude > 0:
                    logger(f"✅ Using separated background music ({len(background_music)/1000:.1f}s)")
                    
                    # Adjust volume to 60% to leave room for vocals
                    background_music = background_music - 4  # Reduce by 4dB
                    
                    # Extend or trim to match video duration
                    if len(background_music) < video_duration_ms:
                        # Loop the background music if it's shorter
                        loops_needed = int(video_duration_ms / len(background_music)) + 1
                        background_music = background_music * loops_needed
                        logger(f"   🔄 Looped background music {loops_needed} times")
                    
                    # Trim to exact duration
                    background_music = background_music[:video_duration_ms]
                    return background_music
                else:
                    logger("⚠️ Separated background music is silent")
            except Exception as e:
                logger(f"⚠️ Error loading background music: {e}")
        
        # Fallback: use low-volume original audio as background
        logger("🔄 No background music available - using silent track")
        logger("   💡 Consider using original audio with reduced volume for better results")
        return AudioSegment.silent(duration=video_duration_ms)