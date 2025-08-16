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
    
    def get_segment_dir(self, segment_index, segment_name):
        """Get directory for processing a specific segment."""
        if not self.current_run_dir:
            raise ValueError("Working directory not set up. Call setup_working_directory first.")
            
        segment_dir = self.current_run_dir / f"segment_{segment_index}_{segment_name}"
        segment_dir.mkdir(exist_ok=True)
        return str(segment_dir)
    
    def save_dubbing_script(self, video_path, dubbing_script, logger):
        """Save the dubbing script to the working directory."""
        import json
        
        try:
            base_name = Path(video_path).stem
            script_filename = f"{base_name}_dubbing_script.json"
            script_path = self.base_working_dir / script_filename
            
            with open(script_path, 'w', encoding='utf-8') as f:
                json.dump(dubbing_script, f, indent=2, ensure_ascii=False)
            
            logger(f"💾 Dubbing script saved: {self.working_dir_name}/{script_filename}")
            logger(f"   📊 Found {len(dubbing_script)} dialogue segments")
            return str(script_path)
            
        except Exception as e:
            logger(f"⚠️ Could not save dubbing script: {e}")
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