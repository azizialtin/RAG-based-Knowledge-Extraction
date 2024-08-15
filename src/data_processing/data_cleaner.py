"""
This module defines all methods needed for cleaning data before using in RAG.
"""

import re
from fastapi import UploadFile
from loguru import logger


class MarkdownCleaner:
    """
    Defines all methods needed for cleaning Markdown files.
    """
    @staticmethod
    def clean_markdown(text: str) -> str:
        """
        Cleans a given Markdown text by performing the following operations:
        1. Removes image references
        2. Collapses multiple consecutive new lines into a maximum of two
        new lines.
        :param: text (str): The Markdown text to be cleaned.
        :return: str: The cleaned Markdown text.
        """
        logger.debug("Removing image references")
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

        logger.debug("Removing multiple new lines (when more than 2)")
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text

    @staticmethod
    def markdown_processing(markdown_file: UploadFile) -> str:
        """
        Reads the content of the provided Markdown file, decodes it,
        and then cleans the text.
        :param: markdown_file: A file-like object containing the Markdown file.
        :return: str: The cleaned Markdown text.
        """

        try:
            # Read the content of the file as a string
            content = markdown_file.file.read().decode("utf-8")
            cleaned_text = MarkdownCleaner.clean_markdown(content)
            return cleaned_text

        except (UnicodeDecodeError, AttributeError) as e:
            logger.error(f"Error reading or cleaning markdown file: {e}")
            raise e
