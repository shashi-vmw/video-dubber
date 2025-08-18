import os
import datetime
from pathlib import Path


class FileManager:
    """Manages working directories and file operations for video dubbing."""
    
    def __init__(self, working_dir_name="working-dir"):
        self.working_dir_name = working_dir_name
        # Handle both absolute and relative paths
        if os.path.isabs(working_dir_name):
            self.base_working_dir = Path(working_dir_name)
        else:
            self.base_working_dir = Path.cwd() / working_dir_name
        self.current_run_dir = None
        
    def setup_working_directory(self, video_path, reuse_path=None):
        """Create a unique working directory for this processing run or use a provided one."""
        if reuse_path:
            self.current_run_dir = Path(reuse_path)
            self.base_working_dir = self.current_run_dir.parent
            return str(self.current_run_dir)

        # Ensure base working directory exists
        self.base_working_dir.mkdir(exist_ok=True)
        
        # Create timestamped run directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(video_path).stem
        run_dir_name = f"{base_name}_{timestamp}"
        self.current_run_dir = self.base_working_dir / run_dir_name
        self.current_run_dir.mkdir(exist_ok=True)
        
        return str(self.current_run_dir)
    
    def setup_output_directory(self, output_path):
        """Create output directory structure and return the absolute path."""
        if os.path.isabs(output_path):
            output_dir = Path(output_path)
        else:
            # For relative paths, create from current working directory
            output_dir = Path.cwd() / output_path
        
        # Create the directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return str(output_dir)
    
    def get_segment_dir(self, segment_index, segment_name):
        """Get directory for processing a specific segment."""
        if not self.current_run_dir:
            raise ValueError("Working directory not set up. Call setup_working_directory first.")
            
        segment_dir = self.current_run_dir / f"segment_{segment_index}_{segment_name}"
        segment_dir.mkdir(exist_ok=True)
        return str(segment_dir)
    
    def save_dubbing_script(self, video_path, dubbing_script, logger):
        """Save the dubbing script to the current run directory."""
        import json
        
        try:
            if not self.current_run_dir:
                raise ValueError("Working directory not set up. Call setup_working_directory first.")
                
            base_name = Path(video_path).stem
            script_filename = f"{base_name}_dubbing_script.json"
            script_path = self.current_run_dir / script_filename
            
            with open(script_path, 'w', encoding='utf-8') as f:
                json.dump(dubbing_script, f, indent=2, ensure_ascii=False)
            
            logger(f"💾 Dubbing script saved: {script_filename}")
            logger(f"   📊 Found {len(dubbing_script)} dialogue segments")
            logger(f"   📁 Location: {script_path}")
            return str(script_path)
            
        except Exception as e:
            logger(f"⚠️ Could not save dubbing script: {e}")
            return None
    
    def save_prompt(self, prompt_content, prompt_type, video_path=None, segment_info=None, logger=None):
        """Save prompt content to the working directory for reference."""
        try:
            if not self.current_run_dir:
                raise ValueError("Working directory not set up. Call setup_working_directory first.")
            
            # Create prompts subdirectory
            prompts_dir = self.current_run_dir / "prompts"
            prompts_dir.mkdir(exist_ok=True)
            
            # Generate filename based on prompt type
            if prompt_type == "dubbing_script":
                base_name = Path(video_path).stem if video_path else "video"
                filename = f"{base_name}_dubbing_script_prompt.txt"
            elif prompt_type == "tts":
                if segment_info:
                    speaker = segment_info.get('speaker_label', 'unknown')
                    segment_num = segment_info.get('segment_number', 'unknown')
                    filename = f"tts_prompt_segment_{segment_num}_{speaker}.txt"
                else:
                    filename = "tts_prompt_sample.txt"
            else:
                filename = f"{prompt_type}_prompt.txt"
            
            prompt_path = prompts_dir / filename
            
            # Write prompt with metadata header
            with open(prompt_path, 'w', encoding='utf-8') as f:
                f.write("# PROMPT CONTENT\n")
                f.write(f"# Type: {prompt_type}\n")
                f.write(f"# Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if video_path:
                    f.write(f"# Video: {Path(video_path).name}\n")
                if segment_info:
                    f.write(f"# Segment: {segment_info}\n")
                f.write("# " + "="*60 + "\n\n")
                f.write(prompt_content)
            
            if logger:
                logger(f"💾 Prompt saved: prompts/{filename}")
            
            return str(prompt_path)
            
        except Exception as e:
            if logger:
                logger(f"❌ Error saving prompt: {e}")
            return None
    
    def get_file_size_gb(self, file_path):
        """Get file size in GB."""
        if not os.path.exists(file_path):
            return 0
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024**3)
    
    def cleanup_temp_files(self, file_paths, logger):
        """Clean up temporary files."""
        cleanup_count = 0
        for file_path in file_paths:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    cleanup_count += 1
                except Exception as e:
                    logger(f"   ⚠️ Could not remove {os.path.basename(file_path)}: {e}")
        
        if cleanup_count > 0:
            logger(f"✅ Cleanup completed ({cleanup_count} temporary files removed)")