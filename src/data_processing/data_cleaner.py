"""
This module defines all methods needed for cleaning data before using it in RAG.
"""

import re


class MarkdownCleaner:
    """
    Defines all methods needed for cleaning Markdown files.
    """
    @staticmethod
    def clean_markdown(text):
        """
        Cleans a given Markdown text by performing the following operations:
        1. Removes image references
        2. Collapses multiple consecutive new lines into a maximum of two new lines.
        :param: text (str): The Markdown text to be cleaned.
        :return: str: The cleaned Markdown text.
        """
        # Remove image references
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        # Remove multiple new lines (when more than 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text

    @staticmethod
    def markdown_processing(markdown_file):
        """
        Reads the content of the provided Markdown file, decodes it,
        and then cleans the text.
        :param: markdown_file: A file-like object containing the Markdown file.
        :return: str: The cleaned Markdown text.
        """

        # Read the content of the file as a string
        content = markdown_file.file.read().decode("utf-8")
        cleaned_text = MarkdownCleaner.clean_markdown(content)

        return cleaned_text
