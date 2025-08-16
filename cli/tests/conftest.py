import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch


@pytest.fixture
def mock_video_processor():
    """Mock VideoProcessor class and its methods."""
    with patch('cli.VideoProcessor') as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        mock_instance.process_video_dubbing.return_value = True
        yield mock_instance


@pytest.fixture
def mock_file_system():
    """Mock file system operations."""
    with patch('pathlib.Path.exists') as mock_exists, \
         patch('pathlib.Path.mkdir') as mock_mkdir, \
         patch('os.path.exists') as mock_os_exists:
        mock_exists.return_value = True
        mock_os_exists.return_value = True
        yield {
            'exists': mock_exists,
            'mkdir': mock_mkdir,
            'os_exists': mock_os_exists
        }


@pytest.fixture
def clean_env():
    """Clean environment variables for testing."""
    env_vars = ['GEMINI_API_KEY', 'GOOGLE_CLOUD_PROJECT', 'GOOGLE_CLOUD_LOCATION']
    old_values = {}
    
    # Store and clear existing values
    for var in env_vars:
        old_values[var] = os.environ.pop(var, None)
    
    yield
    
    # Restore original values
    for var, value in old_values.items():
        if value is not None:
            os.environ[var] = value


@pytest.fixture
def temp_video_file():
    """Create a temporary video file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        f.write(b'fake video content')
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    try:
        os.unlink(temp_path)
    except FileNotFoundError:
        pass


@pytest.fixture
def temp_working_dir():
    """Create a temporary working directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "GEMINI_API_KEY": "test-api-key",
        "USE_VERTEX_AI": False,
        "PROJECT_ID": None,
        "LOCATION": None,
        "INPUT_LANGUAGE": "English",
        "OUTPUT_LANGUAGE": "Spanish",
        "MODEL_NAME": "models/gemini-1.5-pro",
        "TTS_MODEL": "gemini-1.5-pro-preview-tts",
        "COMPRESSION_PROFILE": None,
        "WORKING_DIR": "working-dir",
        "REUSE_PATH": None,
        "STRICT": False,
        "EXTRACTION_ONLY": False,
    }


@pytest.fixture
def mock_logger():
    """Mock logger function."""
    with patch('cli.logger') as mock_log:
        yield mock_log