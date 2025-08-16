# CLI Video Dubber

A command-line interface for the AI-powered video dubbing solution using Google's Gemini models.

## Overview

This CLI tool provides a streamlined way to dub videos from any source language to a target language while preserving original production quality, including background music, emotional tone, and dialogue delivery style.

## Features

- **Audio Separation**: Uses Demucs to separate vocals from background music
- **Multimodal Analysis**: Leverages Gemini 2.5 Pro for comprehensive video analysis
- **Expressive TTS**: Generates natural-sounding dubbed audio with emotion and delivery style
- **Time Synchronization**: Matches dubbed audio timing to original lip movements
- **Flexible Authentication**: Supports both API keys and Vertex AI authentication

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have `ffmpeg` installed on your system for audio/video processing.

3. For development and testing, install test dependencies:
```bash
pip install -r requirements-test.txt
```

## Usage

### Basic Usage

```bash
python cli.py --input-video input.mp4 --output-video output.mp4 --output-language Spanish --gemini-api-key YOUR_API_KEY
```

### Using Vertex AI

```bash
python cli.py --input-video input.mp4 --output-video output.mp4 --output-language Spanish --use-vertex-ai --project-id YOUR_PROJECT --location us-central1
```

### Extraction Only Mode

To extract audio, separate music, and generate script without dubbing:

```bash
python cli.py --input-video input.mp4 --extraction-only
```

### Reusing Previous Work

To continue from a previous run's working directory:

```bash
python cli.py --input-video input.mp4 --output-video output.mp4 --output-language Spanish --reuse working-dir/previous-run
```

## Detailed Examples

### Example 1: Telugu to Hindi Dubbing with Compression
```bash
python cli.py \
--GEMINI_API_KEY YOUR_API_KEY_1 \
--PROJECT_ID your-project-1 \
--input-video ../source-videos/input-video.mov \
--output-video ../output-videos/output-video.mov \
--input-language Telugu \
--output-language Hindi \
--llm-model gemini-2.5-pro \
--tts-model gemini-2.5-pro-preview-tts \
--compress
```
**Purpose**: Dubs a Telugu video to Hindi with automatic compression. Uses the latest Gemini 2.5 Pro models for both analysis and TTS. The `--compress` flag automatically selects an appropriate compression profile for faster processing.

### Example 2: Telugu to English Dubbing (No Compression)
```bash
python cli.py \
--GEMINI_API_KEY YOUR_API_KEY_2 \
--PROJECT_ID your-project-2 \
--input-video ../source-videos/input-video.mov \
--output-video ../output-videos/output-video.mov \
--input-language Telugu \
--output-language English \
--llm-model gemini-2.5-pro \
--tts-model gemini-2.5-pro-preview-tts
```
**Purpose**: Performs high-quality Telugu to English dubbing without compression, maintaining original video quality. Uses different API credentials to distribute load across multiple projects and avoid TTS service limits.

### Example 3: Reusing Previous Processing
```bash
python cli.py \
--GEMINI_API_KEY YOUR_API_KEY_1 \
--PROJECT_ID your-project-1 \
--input-video ../source-videos/input-video.mov \
--output-video ../output-videos/output-video.mov \
--input-language Telugu \
--output-language Hindi \
--llm-model gemini-2.5-pro \
--tts-model gemini-2.5-pro-preview-tts \
--compress \
--reuse ./working-dir/input-video_20250816_132339
```
**Purpose**: Continues dubbing from a previous run's working directory. Skips audio extraction, separation, and script generation steps, directly proceeding to TTS synthesis. Useful for trying different target languages or when previous run was interrupted.

### Example 4: Extraction Only Mode
```bash
python cli.py \
--input-video ../source-videos/input-video.mov \
--output-video ../output-videos/output-video.mov \
--compress \
--extraction-only
```
**Purpose**: Performs only the preparation steps - extracts audio, separates vocals from background music using Demucs, and generates the dubbing script. No API keys required. Useful for preprocessing before actual dubbing or for script review/editing.

**Note**: Multiple API keys and project IDs are used across examples to distribute load and avoid hitting rate limits on the preview TTS service.

## Command Line Options

| Option | Description | Required |
|--------|-------------|----------|
| `--input-video` | Path to input video file | Yes |
| `--output-video` | Path for output video file | Yes (except extraction-only) |
| `--output-language` | Target language for dubbing | Yes (except extraction-only) |
| `--gemini-api-key` | Google API key (or set GEMINI_API_KEY env var) | Yes (unless using Vertex AI) |
| `--use-vertex-ai` | Use Vertex AI authentication | No |
| `--project-id` | Google Cloud project ID | Yes (if using Vertex AI) |
| `--location` | Google Cloud location | Yes (if using Vertex AI) |
| `--input-language` | Original video language | No (default: English) |
| `--llm-model` | LLM model for analysis | No (default: gemini-1.5-pro) |
| `--tts-model` | TTS model for speech synthesis | No (default: gemini-1.5-pro-preview-tts) |
| `--compress` | Video compression profile (360p/720p/1080p) | No |
| `--working-dir` | Directory for temporary files | No (default: working-dir) |
| `--reuse` | Path to previous run's working directory | No |
| `--strict` | Fail immediately on AI generation errors | No |
| `--extraction-only` | Run only extraction steps | No |

## Environment Variables

- `GEMINI_API_KEY`: Your Google API key
- `GOOGLE_CLOUD_PROJECT`: Your Google Cloud project ID
- `GOOGLE_CLOUD_LOCATION`: Your Google Cloud location

## Architecture

The CLI consists of several modules:

- **cli.py**: Main entry point and command-line interface
- **video_processor.py**: Main orchestrator for the dubbing pipeline
- **audio_processor.py**: Handles audio extraction, TTS synthesis, and time correction
- **video_splitter.py**: Manages video segmentation for large files
- **dubbing_script_generator.py**: Generates detailed dubbing scripts using Gemini
- **file_manager.py**: Manages working directories and file operations

## Working Directory Structure

```
working-dir/
├── run_YYYYMMDD_HHMMSS/
│   ├── input_video.*
│   ├── audio.wav
│   ├── separated/
│   │   ├── vocals.wav
│   │   └── no_vocals.wav
│   ├── script.json
│   ├── dubbed_vocals/
│   └── final_output.*
```

## Requirements

- Python 3.8+
- FFmpeg
- Google Cloud SDK (for Vertex AI authentication)
- Internet connection for API calls

## Supported Languages

The tool supports all languages available in Google's Gemini TTS model. Check the Gemini documentation for the most up-to-date list of supported languages.

## Testing

The CLI includes a comprehensive test suite that covers all critical functionality with mocked dependencies.

### Running Tests

Use the test runner script for convenient test execution:

```bash
# Install test dependencies
python run_tests.py install

# Run all tests
python run_tests.py all

# Run specific test categories
python run_tests.py unit           # Unit tests only
python run_tests.py integration    # Integration tests only
python run_tests.py validation     # CLI validation tests
python run_tests.py config         # Configuration tests
python run_tests.py main           # Main CLI function tests

# Run tests with coverage report
python run_tests.py coverage
```

### Test Structure

- **Unit Tests**: Test individual components and validation logic
- **Integration Tests**: Test complete CLI workflows with mocked VideoProcessor
- **Validation Tests**: Test CLI argument parsing and validation
- **Configuration Tests**: Test configuration object creation

### What's Tested

- ✅ All CLI argument combinations and validation
- ✅ Authentication methods (API key, Vertex AI, environment variables)
- ✅ Conflicting flag detection
- ✅ Configuration object creation and defaults
- ✅ Full dubbing workflow with mocked dependencies
- ✅ Error handling and exit codes
- ✅ Extraction-only mode
- ✅ Reuse mode functionality
- ✅ Compression profile validation
- ✅ Multiple language combinations

### Mocked Dependencies

Tests use mocks for:
- VideoProcessor and all its methods
- File system operations
- Environment variables
- Google API calls
- FFmpeg operations

This ensures tests run quickly and reliably without requiring actual video files or API credentials.