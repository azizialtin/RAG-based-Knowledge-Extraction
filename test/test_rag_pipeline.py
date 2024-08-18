# pylint:disable = missing-function-docstring, missing-module-docstring, missing-class-docstring, invalid-name
import unittest
from unittest.mock import MagicMock, patch
from langchain_core.vectorstores import VectorStoreRetriever
from src.models import RerankingConfigs, GenerationConfigs
from src.rag.rag_pipeline import RAGPipeline


class TestRAGPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = RAGPipeline()

    @patch('src.rag.rag_pipeline.ContextualCompressionRetriever')
    @patch('src.rag.rag_pipeline.FlashrankRerank')
    def test_initialize_reranker(self, mock_FlashrankRerank, mock_ContextualCompressionRetriever):
        mock_base_retriever = MagicMock(VectorStoreRetriever)
        reranker_configs = RerankingConfigs(model_id="test-model", top_k=5)

        self.pipeline.initialize_reranker(mock_base_retriever, reranker_configs)

        mock_FlashrankRerank.assert_called_once_with(model="test-model", top_n=5)
        mock_ContextualCompressionRetriever.assert_called_once()

    @patch('src.rag.rag_pipeline.Ollama')
    @patch('src.rag.rag_pipeline.is_model_available')
    def test_initialize_llm(self, mock_is_model_available, mock_Ollama):
        gen_configs = GenerationConfigs(model_id="test-model")

        self.pipeline.initialize_llm(gen_configs)

        mock_is_model_available.assert_called_once_with("test-model")
        mock_Ollama.assert_called_once_with(model="test-model")
        self.assertIsNotNone(self.pipeline.llm)


if __name__ == "__main__":
    unittest.main()
