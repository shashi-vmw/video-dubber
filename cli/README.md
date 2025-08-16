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