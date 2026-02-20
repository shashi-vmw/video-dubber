# Gemini Video Dubber

An AI-powered video dubbing and translation application using Google Gemini and Vertex AI.

## Features
- **Automatic Translation**: Translates video audio from source language to target language.
- **Syllable Ratio Optimization**: Adapts translations to fit the original speech duration.
- **Adaptive A/V Sync**: Dynamically slows down or speeds up video segments to maintain lip-sync.
- **Vocal Separation**: Uses Demucs to preserve original background score.
- **Expressive Dubbing**: Captures sighs, laughs, and emotional nuances using Gemini TTS.
- **Interactive UI**: 2-step process to review scripts and manually assign voices.

## Usage
1.  Set up Google Cloud Platform (GCP) credentials.
2.  Install dependencies: `pip install -r CloudRun/requirements.txt`
3.  Run the app: `streamlit run app_local.py`

## Cloud Deployment
The `CloudRun/` directory contains a Flask API and Dockerfile for deploying the dubbing engine to Google Cloud Run.
