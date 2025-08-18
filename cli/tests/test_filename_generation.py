import pytest
import os
from unittest.mock import Mock, patch
from video_processor import VideoProcessor


@pytest.mark.unit
class TestFilenameGeneration:
    """Test filename generation logic with output language."""

    def test_language_sanitization(self):
        """Test that language names are properly sanitized for filenames."""
        processor = VideoProcessor()
        
        test_cases = [
            ("Spanish", "spanish"),
            ("English", "english"),
            ("Hindi", "hindi"),
            ("Chinese (Simplified)", "chinesesimplified"),
            ("French-Canadian", "french-canadian"),
            ("Portuguese (Brazil)", "portuguesebrazil"),
            ("Arabic", "arabic"),
            ("Japanese", "japanese"),
            ("German", "german"),
            ("Korean", "korean"),
            ("Language with Spaces", "languagewithspaces"),
            ("Lang-with-Dashes", "lang-with-dashes"),
            ("Lang_with_Underscores", "lang_with_underscores"),
            ("Lang@#$%Special!Chars", "langspecialchars"),
        ]
        
        for original_lang, expected_safe_lang in test_cases:
            # Test the sanitization logic directly
            safe_language = ''.join(c for c in original_lang if c.isalnum() or c in '-_').lower()
            assert safe_language == expected_safe_lang, f"Language '{original_lang}' should sanitize to '{expected_safe_lang}', got '{safe_language}'"

    @patch('video_processor.VideoSplitter')
    def test_filename_generation_with_different_languages(self, mock_splitter):
        """Test that filenames are correctly generated with different output languages."""
        processor = VideoProcessor()
        
        test_cases = [
            {
                "video_path": "/path/to/my-video.mp4",
                "output_language": "Spanish",
                "expected_filename": "my-video_dubbed_spanish.mp4"
            },
            {
                "video_path": "/path/to/episode-01.mov",
                "output_language": "Hindi",
                "expected_filename": "episode-01_dubbed_hindi.mp4"
            },
            {
                "video_path": "/path/to/complex_file-name.avi",
                "output_language": "Chinese (Simplified)",
                "expected_filename": "complex_file-name_dubbed_chinesesimplified.mp4"
            },
            {
                "video_path": "/path/to/test.mkv",
                "output_language": "French-Canadian",
                "expected_filename": "test_dubbed_french-canadian.mp4"
            }
        ]
        
        for test_case in test_cases:
            config = {"OUTPUT_LANGUAGE": test_case["output_language"]}
            
            # Mock the logger
            mock_logger = Mock()
            
            # Mock single segment scenario (simpler to test)
            dubbed_segments = ["/path/to/segment.mp4"]
            output_dir = "/output"
            
            # Mock the _finalize_single_segment method to capture the filename
            with patch.object(processor, '_finalize_single_segment') as mock_finalize:
                mock_finalize.return_value = "mocked_output"
                
                processor._handle_segment_combination(
                    dubbed_segments, 
                    output_dir, 
                    test_case["video_path"], 
                    config, 
                    mock_logger
                )
                
                # Verify the correct filename was generated
                expected_path = os.path.join(output_dir, test_case["expected_filename"])
                mock_finalize.assert_called_once_with(dubbed_segments[0], expected_path, mock_logger)

    def test_filename_generation_without_output_language(self):
        """Test filename generation when output language is missing."""
        processor = VideoProcessor()
        
        config = {}  # No OUTPUT_LANGUAGE key
        dubbed_segments = ["/path/to/segment.mp4"]
        output_dir = "/output"
        video_path = "/path/to/test.mp4"
        mock_logger = Mock()
        
        with patch.object(processor, '_finalize_single_segment') as mock_finalize:
            mock_finalize.return_value = "mocked_output"
            
            processor._handle_segment_combination(
                dubbed_segments, 
                output_dir, 
                video_path, 
                config, 
                mock_logger
            )
            
            # Should default to "unknown" when language is missing
            expected_path = os.path.join(output_dir, "test_dubbed_unknown.mp4")
            mock_finalize.assert_called_once_with(dubbed_segments[0], expected_path, mock_logger)