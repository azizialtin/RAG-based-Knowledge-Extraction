# pylint:disable = missing-function-docstring, missing-module-docstring, missing-class-docstring, invalid-name, too-few-public-methods
import unittest
from fastapi import HTTPException
from langchain_core.documents import Document
from src.utils import check_json, semantic_split_sections, paragraph_split_sections


class MockEmbeddingModel:
    """Mock embedding model for testing purposes."""
    def encode(self, texts):
        return [list(range(768))] * len(texts)


class TestUtils(unittest.TestCase):

    def test_check_json_invalid(self):
        invalid_json = '{"index_name": "test_index", ' \
                       '"retrieval_configs": {' \
                           '"model_id": "test_model"' \
                       '}, ' \
                       '"generation_configs": {' \
                           '"model_id": "test_gen_model", ' \
                           '"max_tokens": 50}' \
                       '} '
        with self.assertRaises(HTTPException) as context:
            check_json(index_config=invalid_json)
        self.assertEqual(context.exception.status_code, 422)

    def test_semantic_split_sections_no_split(self):
        embedding_model = MockEmbeddingModel()
        documents = [Document(
            page_content="This is a short document.",
            metadata={"source": "test_source"}
        )]

        split_documents = semantic_split_sections(documents, embedding_model)
        self.assertEqual(len(split_documents), 1)
        self.assertEqual(split_documents[0].page_content, "This is a short document.")

    def test_paragraph_split_sections_single_line(self):
        documents = [Document(
            page_content="Single line without paragraph split.",
            metadata={"source": "test_source"}
        )]

        split_documents = paragraph_split_sections(documents)
        self.assertEqual(len(split_documents), 1)
        self.assertEqual(split_documents[0].page_content, "Single line without paragraph split.")


if __name__ == '__main__':
    unittest.main()
