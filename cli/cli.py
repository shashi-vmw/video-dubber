import click
import os
from video_processor import VideoProcessor

def logger(message):
    """A simple logger that prints messages to the console."""
    click.echo(message)

@click.command()
@click.option("--input-video", required=True, type=click.Path(exists=True), help="Path to the input video file.")
@click.option("--output-path", required=True, type=click.Path(), help="Relative path to the output directory where dubbed videos will be saved.")
@click.option("--output-language", help="The language to dub the video into (required unless in extraction-only mode).")
@click.option("--gemini-api-key", envvar="GEMINI_API_KEY", help="Your Google API key. Can also be set via the GEMINI_API_KEY environment variable.")
@click.option("--use-vertex-ai", is_flag=True, help="Use Vertex AI for authentication.")
@click.option("--project-id", envvar="GOOGLE_CLOUD_PROJECT", help="Your Google Cloud project ID (required for Vertex AI). Can also be set via the GOOGLE_CLOUD_PROJECT environment variable.")
@click.option("--location", envvar="GOOGLE_CLOUD_LOCATION", help="Your Google Cloud location (required for Vertex AI). Can also be set via the GOOGLE_CLOUD_LOCATION environment variable.")
@click.option("--input-language", default="English", show_default=True, help="The original language of the video.")
@click.option("--llm-model", default="gemini-1.5-pro", show_default=True, help="The LLM model to use for translation.")
@click.option("--tts-model", default="gemini-1.5-pro-preview-tts", show_default=True, help="The TTS model to use for speech synthesis.")
@click.option("--compress", "compression_profile", type=click.Choice(['360p', '720p', '1080p'], case_sensitive=False), help="Compress video to a profile for faster processing. If not provided, no compression is used.")
@click.option("--working-dir", default="working-dir", show_default=True, help="Directory for temporary files and outputs.")
@click.option("--reuse", "reuse_path", type=click.Path(exists=True, file_okay=False, dir_okay=True), help="Path to a previous run's working directory to reuse its artifacts.")
@click.option("--strict", is_flag=True, help="Fail immediately if any AI generation step (TTS, translation) fails.")
@click.option("--extraction-only", is_flag=True, help="Run only the extraction steps (audio, music separation, script) and then stop.")
def main(input_video, output_path, output_language, gemini_api_key, use_vertex_ai, project_id, location, input_language, llm_model, tts_model, compression_profile, working_dir, reuse_path, strict, extraction_only):
    """
    A command-line tool to dub videos using the Gemini AI API.
    """
    if not extraction_only and not output_language:
        raise click.UsageError("Missing option '--output-language'. This is required for the full dubbing process.")

    if reuse_path and strict:
        raise click.UsageError("--reuse and --strict flags cannot be used together.")
    if reuse_path and extraction_only:
        raise click.UsageError("--reuse and --extraction-only flags cannot be used together.")

    # API key is not needed for extraction-only mode
    if not extraction_only:
        if not use_vertex_ai and not gemini_api_key:
            raise click.UsageError("Please provide a Gemini API Key (using --gemini-api-key or the GEMINI_API_KEY environment variable) or select '--use-vertex-ai'.")
        
        if use_vertex_ai and (not project_id or not location):
            raise click.UsageError("When using --use-vertex-ai, you must provide a Project ID and Location (using command-line options or environment variables).")

    config = {
        "GEMINI_API_KEY": gemini_api_key,
        "USE_VERTEX_AI": use_vertex_ai,
        "PROJECT_ID": project_id,
        "LOCATION": location,
        "INPUT_LANGUAGE": input_language,
        "OUTPUT_LANGUAGE": output_language,
        "MODEL_NAME": f"models/{llm_model}",
        "TTS_MODEL": tts_model,
        "COMPRESSION_PROFILE": compression_profile,
        "WORKING_DIR": working_dir,
        "REUSE_PATH": reuse_path,
        "STRICT": strict,
        "EXTRACTION_ONLY": extraction_only,
    }

    # Initialize the video processor
    processor = VideoProcessor(working_dir)
    
    logger("🚀 Starting video dubbing process...")
    
    try:
        result = processor.process_video_dubbing(input_video, output_path, config, logger)
        
        if result:
            logger("✅ Video dubbing process completed successfully!")
        else:
            logger("❌ Video dubbing process failed!")
            exit(1)
    except Exception as e:
        logger("❌ Video dubbing process failed!")
        exit(1)

if __name__ == "__main__":
    main()