import pytest
from click.testing import CliRunner
from cli import main


@pytest.mark.unit
class TestConfigCreation:
    """Test configuration object creation from CLI arguments."""

    def test_default_config_values(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test that default configuration values are set correctly."""
        runner = CliRunner()
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-path', 'output',
            '--output-language', 'Spanish',
            '--gemini-api-key', 'test-key'
        ])
        
        assert result.exit_code == 0
        
        call_args = mock_video_processor.process_video_dubbing.call_args
        config = call_args[0][2]
        
        # Test default values
        assert config['INPUT_LANGUAGE'] == 'English'
        assert config['MODEL_NAME'] == 'models/gemini-1.5-pro'
        assert config['TTS_MODEL'] == 'gemini-1.5-pro-preview-tts'
        assert config['WORKING_DIR'] == 'working-dir'
        assert config['USE_VERTEX_AI'] is False
        assert config['STRICT'] is False
        assert config['EXTRACTION_ONLY'] is False
        assert config['COMPRESSION_PROFILE'] is None
        assert config['REUSE_PATH'] is None
        assert config['PROJECT_ID'] is None
        assert config['LOCATION'] is None

    def test_config_with_all_options(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test configuration creation with all possible non-conflicting options."""
        runner = CliRunner()
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-path', 'output',
            '--output-language', 'French',
            '--gemini-api-key', 'custom-api-key',
            '--project-id', 'custom-project',
            '--location', 'us-west1',
            '--input-language', 'German',
            '--llm-model', 'gemini-2.5-pro',
            '--tts-model', 'gemini-2.5-pro-preview-tts',
            '--compress', '1080p',
            '--working-dir', 'custom-working-dir'
        ])
        
        assert result.exit_code == 0
        
        call_args = mock_video_processor.process_video_dubbing.call_args
        config = call_args[0][2]
        
        # Test all custom values
        assert config['GEMINI_API_KEY'] == 'custom-api-key'
        assert config['PROJECT_ID'] == 'custom-project'
        assert config['LOCATION'] == 'us-west1'
        assert config['INPUT_LANGUAGE'] == 'German'
        assert config['OUTPUT_LANGUAGE'] == 'French'
        assert config['MODEL_NAME'] == 'models/gemini-2.5-pro'
        assert config['TTS_MODEL'] == 'gemini-2.5-pro-preview-tts'
        assert config['COMPRESSION_PROFILE'] == '1080p'
        assert config['WORKING_DIR'] == 'custom-working-dir'
        assert config['REUSE_PATH'] is None  # Not set in this test
        assert config['USE_VERTEX_AI'] is False  # Not set
        assert config['STRICT'] is False  # Not set in this test
        assert config['EXTRACTION_ONLY'] is False

    def test_vertex_ai_config(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test configuration with Vertex AI enabled."""
        runner = CliRunner()
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-path', 'output',
            '--output-language', 'Japanese',
            '--use-vertex-ai',
            '--project-id', 'vertex-project',
            '--location', 'asia-northeast1'
        ])
        
        assert result.exit_code == 0
        
        call_args = mock_video_processor.process_video_dubbing.call_args
        config = call_args[0][2]
        
        assert config['USE_VERTEX_AI'] is True
        assert config['PROJECT_ID'] == 'vertex-project'
        assert config['LOCATION'] == 'asia-northeast1'
        assert config['GEMINI_API_KEY'] is None

    def test_extraction_only_config(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test configuration for extraction-only mode."""
        runner = CliRunner()
        
        result = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-path', 'output',
            '--extraction-only',
            '--compress', '720p'
        ])
        
        assert result.exit_code == 0
        
        call_args = mock_video_processor.process_video_dubbing.call_args
        config = call_args[0][2]
        
        assert config['EXTRACTION_ONLY'] is True
        assert config['COMPRESSION_PROFILE'] == '720p'
        assert config['OUTPUT_LANGUAGE'] is None
        assert config['GEMINI_API_KEY'] is None
        assert config['USE_VERTEX_AI'] is False

    def test_model_name_formatting(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test that model names are formatted with 'models/' prefix."""
        runner = CliRunner()
        
        custom_models = [
            'gemini-1.5-pro',
            'gemini-2.5-pro',
            'custom-model'
        ]
        
        for model in custom_models:
            result = runner.invoke(main, [
                '--input-video', temp_video_file,
                '--output-path', 'output',
                '--output-language', 'Spanish',
                '--gemini-api-key', 'test-key',
                '--llm-model', model
            ])
            
            assert result.exit_code == 0
            
            call_args = mock_video_processor.process_video_dubbing.call_args
            config = call_args[0][2]
            
            assert config['MODEL_NAME'] == f'models/{model}'

    def test_boolean_flags(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test that boolean flags are properly set in configuration."""
        runner = CliRunner()
        
        # Test individual boolean flags
        test_cases = [
            (['--use-vertex-ai', '--project-id', 'test', '--location', 'us-central1'], 'USE_VERTEX_AI', True),
            (['--strict', '--gemini-api-key', 'test'], 'STRICT', True),
            (['--extraction-only'], 'EXTRACTION_ONLY', True),
        ]
        
        for args, config_key, expected_value in test_cases:
            base_args = [
                '--input-video', temp_video_file,
                '--output-path', 'output'
            ]
            if config_key != 'EXTRACTION_ONLY':
                base_args.extend(['--output-language', 'Spanish'])
            
            result = runner.invoke(main, base_args + args)
            
            assert result.exit_code == 0
            
            call_args = mock_video_processor.process_video_dubbing.call_args
            config = call_args[0][2]
            
            assert config[config_key] == expected_value

    def test_compression_profile_case_insensitive(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test that compression profiles are case-insensitive."""
        runner = CliRunner()
        
        profiles = ['360p', '360P', '720p', '720P', '1080p', '1080P']
        
        for profile in profiles:
            result = runner.invoke(main, [
                '--input-video', temp_video_file,
                '--output-path', 'output',
                '--output-language', 'Spanish',
                '--gemini-api-key', 'test-key',
                '--compress', profile
            ])
            
            assert result.exit_code == 0
            
            call_args = mock_video_processor.process_video_dubbing.call_args
            config = call_args[0][2]
            
            # Click normalizes to lowercase
            assert config['COMPRESSION_PROFILE'] == profile.lower()

    def test_config_immutability_between_calls(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test that configuration doesn't leak between different CLI invocations."""
        runner = CliRunner()
        
        # First call with specific config
        result1 = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-path', 'output1',
            '--output-language', 'Spanish',
            '--gemini-api-key', 'key1',
            '--compress', '720p',
            '--strict'
        ])
        
        assert result1.exit_code == 0
        first_call_args = mock_video_processor.process_video_dubbing.call_args
        first_config = first_call_args[0][2]
        
        # Second call with different config
        result2 = runner.invoke(main, [
            '--input-video', temp_video_file,
            '--output-path', 'output2',
            '--output-language', 'French',
            '--gemini-api-key', 'key2'
        ])
        
        assert result2.exit_code == 0
        second_call_args = mock_video_processor.process_video_dubbing.call_args
        second_config = second_call_args[0][2]
        
        # Verify configs are different
        assert first_config['OUTPUT_LANGUAGE'] == 'Spanish'
        assert second_config['OUTPUT_LANGUAGE'] == 'French'
        assert first_config['GEMINI_API_KEY'] == 'key1'
        assert second_config['GEMINI_API_KEY'] == 'key2'
        assert first_config['COMPRESSION_PROFILE'] == '720p'
        assert second_config['COMPRESSION_PROFILE'] is None
        assert first_config['STRICT'] is True
        assert second_config['STRICT'] is False

    def test_config_parameter_order_independence(self, mock_video_processor, mock_file_system, temp_video_file, clean_env):
        """Test that parameter order doesn't affect configuration."""
        runner = CliRunner()
        
        # Same parameters in different order
        args1 = [
            '--input-video', temp_video_file,
            '--output-path', 'output',
            '--output-language', 'Spanish',
            '--gemini-api-key', 'test-key',
            '--compress', '720p',
            '--strict'
        ]
        
        args2 = [
            '--strict',
            '--compress', '720p',
            '--gemini-api-key', 'test-key',
            '--output-language', 'Spanish',
            '--output-path', 'output',
            '--input-video', temp_video_file
        ]
        
        result1 = runner.invoke(main, args1)
        assert result1.exit_code == 0
        config1 = mock_video_processor.process_video_dubbing.call_args[0][2]
        
        result2 = runner.invoke(main, args2)
        assert result2.exit_code == 0
        config2 = mock_video_processor.process_video_dubbing.call_args[0][2]
        
        # Configs should be identical
        assert config1 == config2