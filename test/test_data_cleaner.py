# pylint:disable = missing-function-docstring, missing-module-docstring, missing-class-docstring, invalid-name
import unittest
from io import BytesIO
from fastapi import UploadFile
from src.data_processing import MarkdownCleaner


class TestMarkdownCleaner(unittest.TestCase):

    def test_clean_markdown_removes_image_references(self):
        markdown_text = "This is a test ![image](http://example.com/image.png) with an image."
        expected_output = "This is a test  with an image."
        cleaned_text = MarkdownCleaner.clean_markdown(markdown_text)
        self.assertEqual(cleaned_text, expected_output)

    def test_clean_markdown_collapses_multiple_newlines(self):
        markdown_text = "Line 1\n\n\n\nLine 2"
        expected_output = "Line 1\n\nLine 2"
        cleaned_text = MarkdownCleaner.clean_markdown(markdown_text)
        self.assertEqual(cleaned_text, expected_output)

    def test_clean_markdown_handles_no_images_or_extra_newlines(self):
        markdown_text = "This is a test with no images or extra newlines."
        expected_output = "This is a test with no images or extra newlines."
        cleaned_text = MarkdownCleaner.clean_markdown(markdown_text)
        self.assertEqual(cleaned_text, expected_output)

    def test_markdown_processing(self):
        content = "This is a test ![image](http://example.com/image.png)\n\n\nLine 2"
        expected_output = "This is a test \n\nLine 2"

        # Mocking the UploadFile with BytesIO content
        file_like = BytesIO(content.encode("utf-8"))
        upload_file = UploadFile(filename="test.md", file=file_like)

        cleaned_text = MarkdownCleaner.markdown_processing(upload_file)
        self.assertEqual(cleaned_text, expected_output)

    def test_markdown_processing_unicode_error(self):
        content = b"\x80\x81\x82"  # Invalid UTF-8 bytes
        file_like = BytesIO(content)
        upload_file = UploadFile(filename="test.md", file=file_like)

        with self.assertRaises(UnicodeDecodeError):
            MarkdownCleaner.markdown_processing(upload_file)

    def test_markdown_processing_attribute_error(self):
        # Passing None instead of UploadFile to trigger an AttributeError
        with self.assertRaises(AttributeError):
            MarkdownCleaner.markdown_processing(None)


if __name__ == '__main__':
    unittest.main()
