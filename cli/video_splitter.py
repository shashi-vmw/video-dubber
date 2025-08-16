import os
import time
import math
import subprocess
from moviepy import VideoFileClip
from file_manager import FileManager


class VideoSplitter:
    """Handles video compression and splitting for large files."""
    
    def __init__(self, max_size_gb=1.8):
        self.max_size_gb = max_size_gb
        self.file_manager = FileManager()
    
    def compress_video_for_testing(self, input_path, output_dir, profile, logger):
        """Compress video aggressively for testing purposes based on a selected profile."""
        logger(f"🗜️ Starting video compression for testing (Profile: {profile})...")
        
        profiles = {
            '360p': {'vf': 'scale=640:360', 'vb': '200k', 'ab': '64k'},
            '720p': {'vf': 'scale=1280:720', 'vb': '1M', 'ab': '128k'},
            '1080p': {'vf': 'scale=1920:1080', 'vb': '2.5M', 'ab': '192k'}
        }
        
        if profile not in profiles:
            logger(f"❌ Invalid compression profile: {profile}. Valid options are {list(profiles.keys())}.")
            return None
            
        settings = profiles[profile]
        
        try:
            # Validate input file
            if not os.path.exists(input_path):
                logger(f"❌ Input video file not found: {input_path}")
                return None
                
            original_size = self.file_manager.get_file_size_gb(input_path)
            logger(f"   📊 Original file size: {original_size:.2f} GB")
            
            # Get video info first
            logger(f"   📹 Analyzing video properties...")
            self._log_video_properties(input_path, logger)
            
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            compressed_path = os.path.join(output_dir, f"{base_name}_compressed_{profile}.mp4")
            
            logger(f"   🎯 Target: {profile}, {settings['vb']} video bitrate, {settings['ab']} audio bitrate")
            logger(f"   📁 Output: {os.path.basename(compressed_path)}")
            
            # Compression command with aspect ratio preservation and audio channel mapping
            # Extract width and height from the scale filter for proper padding
            if ':' in settings['vf']:
                scale_parts = settings['vf'].replace('scale=', '').split(':')
                target_width, target_height = scale_parts[0], scale_parts[1]
                padding_filter = f"{settings['vf']}:force_original_aspect_ratio=decrease,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2"
            else:
                # Fallback for unexpected scale format
                padding_filter = settings['vf']
            
            cmd = [
                "ffmpeg", "-i", input_path,
                "-vf", padding_filter,
                "-b:v", settings['vb'],
                "-b:a", settings['ab'],
                "-r", "24",
                "-preset", "ultrafast",
                "-movflags", "+faststart",
                "-pix_fmt", "yuv420p",
                "-map", "0:v:0",
                "-map", "0:a",
                "-c:a", "aac",
                compressed_path, "-y"
            ]
            
            return self._execute_compression(cmd, input_path, compressed_path, logger)
            
        except Exception as e:
            logger(f"❌ Unexpected error during compression: {str(e)}")
            return None
    
    def split_video_by_size(self, video_path, output_dir, logger):
        """Split video into chunks based on file size limit."""
        logger(f"📹 Analyzing video for splitting...")
        
        try:
            if not os.path.exists(video_path):
                logger(f"❌ Input video file not found: {video_path}")
                return None
                
            logger(f"   📁 Input file: {os.path.basename(video_path)}")
            logger(f"   📏 Size limit: {self.max_size_gb:.1f} GB per chunk")
            
            # Get video information
            video_info = self._get_video_info(video_path, logger)
            if not video_info:
                return None
                
            file_size_gb = self.file_manager.get_file_size_gb(video_path)
            
            if file_size_gb <= self.max_size_gb:
                logger(f"✅ Video size ({file_size_gb:.2f} GB) is within limit ({self.max_size_gb:.1f} GB). No splitting needed.")
                return [video_path]
            
            return self._create_video_chunks(video_path, output_dir, video_info, file_size_gb, logger)
            
        except Exception as e:
            logger(f"❌ Error during video splitting: {str(e)}")
            return None
    
    def combine_dubbed_segments(self, segment_paths, output_path, logger):
        """Combine multiple dubbed video segments into a single file."""
        logger(f"🔗 Combining {len(segment_paths)} dubbed segments...")
        
        if len(segment_paths) == 1:
            # If only one segment, just copy it
            import shutil
            shutil.copy2(segment_paths[0], output_path)
            logger("✅ Single segment copied as final output")
            return output_path
        
        # Create a temporary file list for ffmpeg
        temp_dir = os.path.dirname(output_path)
        filelist_path = os.path.join(temp_dir, "segments_list.txt")
        
        try:
            with open(filelist_path, 'w') as f:
                for segment_path in segment_paths:
                    f.write(f"file '{os.path.abspath(segment_path)}'\n")
            
            # Use ffmpeg to concatenate videos
            cmd = [
                "ffmpeg", "-f", "concat", "-safe", "0",
                "-i", filelist_path,
                "-c", "copy",  # Copy without re-encoding
                output_path, "-y"
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger(f"✅ Successfully combined segments into: {os.path.basename(output_path)}")
            
            # Clean up
            os.remove(filelist_path)
            for segment_path in segment_paths:
                os.remove(segment_path)
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger(f"❌ Failed to combine segments: {e}")
            if os.path.exists(filelist_path):
                os.remove(filelist_path)
            return None
        except Exception as e:
            logger(f"❌ Error during segment combination: {e}")
            return None
    
    def _log_video_properties(self, input_path, logger):
        """Log original video properties."""
        try:
            info_cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", input_path]
            info_result = subprocess.run(info_cmd, capture_output=True, text=True, check=True)
            import json
            video_info = json.loads(info_result.stdout)
            
            for stream in video_info.get('streams', []):
                if stream.get('codec_type') == 'video':
                    width = stream.get('width', 'unknown')
                    height = stream.get('height', 'unknown')
                    fps = stream.get('r_frame_rate', 'unknown')
                    logger(f"   📏 Original resolution: {width}x{height}, FPS: {fps}")
                    break
        except Exception as e:
            logger(f"   ⚠️ Could not analyze video properties: {e}")
    
    def _execute_compression(self, cmd, input_path, compressed_path, logger):
        """Execute the compression command and return results."""
        logger(f"   ⚡ Starting compression (this may take a few minutes)...")
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        
        if result.returncode != 0:
            logger(f"❌ FFmpeg compression failed!")
            logger(f"   💬 FFmpeg stderr: {result.stderr}")
            return None
        
        if not os.path.exists(compressed_path):
            logger(f"❌ Compressed file was not created: {compressed_path}")
            return None
            
        # Calculate and report results
        original_size = self.file_manager.get_file_size_gb(input_path)
        compressed_size = self.file_manager.get_file_size_gb(compressed_path)
        reduction = ((original_size - compressed_size) / original_size) * 100
        compression_time = end_time - start_time
        
        logger(f"✅ Compression completed successfully!")
        logger(f"   📊 Results:")
        logger(f"      • Original: {original_size:.2f} GB")
        logger(f"      • Compressed: {compressed_size:.2f} GB")
        logger(f"      • Size reduction: {reduction:.1f}%")
        logger(f"      • Compression time: {compression_time:.1f} seconds")
        
        return compressed_path
    
    def _get_video_info(self, video_path, logger):
        """Get video information using MoviePy."""
        try:
            with VideoFileClip(video_path) as clip:
                return {
                    'duration': clip.duration,
                    'fps': clip.fps if hasattr(clip, 'fps') else 'unknown',
                    'resolution': f"{clip.size[0]}x{clip.size[1]}" if hasattr(clip, 'size') else 'unknown'
                }
        except Exception as e:
            logger(f"❌ Could not get video info: {e}")
            return None
    
    def _create_video_chunks(self, video_path, output_dir, video_info, file_size_gb, logger):
        """Create video chunks based on size limits."""
        # Calculate number of splits needed
        num_splits = math.ceil(file_size_gb / self.max_size_gb)
        chunk_duration = video_info['duration'] / num_splits
        
        logger(f"🔄 Video exceeds size limit - splitting required:")
        logger(f"   📊 Split plan:")
        logger(f"      • Current size: {file_size_gb:.2f} GB")
        logger(f"      • Max per chunk: {self.max_size_gb:.1f} GB")
        logger(f"      • Number of chunks: {num_splits}")
        logger(f"      • Duration per chunk: ~{chunk_duration/60:.1f} minutes")
        
        chunk_paths = []
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        logger(f"   🎬 Creating {num_splits} video chunks...")
        
        for i in range(num_splits):
            start_time = i * chunk_duration
            end_time = min((i + 1) * chunk_duration, video_info['duration'])
            actual_duration = end_time - start_time
            chunk_path = os.path.join(output_dir, f"{base_name}_chunk_{i+1:02d}.mp4")
            
            logger(f"   ✂️ Chunk {i+1}/{num_splits}:")
            logger(f"      • Time: {start_time/60:.1f}min → {end_time/60:.1f}min ({actual_duration/60:.1f}min duration)")
            
            if self._create_single_chunk(video_path, chunk_path, start_time, actual_duration, logger):
                chunk_paths.append(chunk_path)
            else:
                return None
        
        # Report results
        total_chunks_size = sum(self.file_manager.get_file_size_gb(chunk) for chunk in chunk_paths)
        logger(f"✅ Successfully split video:")
        logger(f"   📊 Summary:")
        logger(f"      • Original: {file_size_gb:.2f} GB")
        logger(f"      • Total chunks: {total_chunks_size:.2f} GB ({len(chunk_paths)} files)")
        
        return chunk_paths
    
    def _create_single_chunk(self, video_path, chunk_path, start_time, duration, logger):
        """Create a single video chunk."""
        cmd = [
            "ffmpeg", "-i", video_path,
            "-ss", str(start_time),
            "-t", str(duration),
            "-c", "copy",  # Copy without re-encoding for speed
            "-avoid_negative_ts", "make_zero",
            "-v", "warning",  # Reduce ffmpeg verbosity
            chunk_path, "-y"
        ]
        
        try:
            split_start_time = time.time()
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            split_end_time = time.time()
            split_duration = split_end_time - split_start_time
            
            if os.path.exists(chunk_path):
                chunk_size = self.file_manager.get_file_size_gb(chunk_path)
                logger(f"      ✅ Created in {split_duration:.1f}s (Size: {chunk_size:.2f} GB)")
                return True
            else:
                logger(f"      ❌ Chunk file was not created")
                return False
                
        except subprocess.CalledProcessError as e:
            logger(f"      ❌ FFmpeg failed to create chunk")
            logger(f"         💬 Error: {e}")
            return False
        except Exception as e:
            logger(f"      ❌ Unexpected error creating chunk: {str(e)}")
            return False