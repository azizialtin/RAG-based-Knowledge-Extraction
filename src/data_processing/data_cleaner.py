import re


class Markdown_Cleaner:
    @staticmethod
    def clean_markdown(text):
        # Remove image references
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        # Remove multiple new lines (when more than 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text

    @staticmethod
    def markdown_processing(markdown_file):
        # Read the content of the file as a string
        content = markdown_file.file.read().decode("utf-8")
        cleaned_text = Markdown_Cleaner.clean_markdown(content)


        return cleaned_text
