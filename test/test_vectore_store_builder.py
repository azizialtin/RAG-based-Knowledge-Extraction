# pylint:disable = missing-function-docstring, missing-module-docstring, missing-class-docstring, invalid-name
import unittest
from unittest.mock import patch, MagicMock
from src.vector_store.vector_store_builder import VectorStoreBuilder


class TestVectorStoreBuilder(unittest.TestCase):

    @patch('src.vector_store.vector_store_builder.HuggingFaceEmbeddings')
    def test_create_embedding(self, mock_hf_embeddings):
        """
        Test that the HuggingFace embedding is correctly created.
        """
        mock_hf_embeddings.return_value = MagicMock()
        embedding_model_id = "test-model"

        result = VectorStoreBuilder._create_embedding(embedding_model_id)

        mock_hf_embeddings.assert_called_once_with(
            model_name=embedding_model_id,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False},
            show_progress=True
        )
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
