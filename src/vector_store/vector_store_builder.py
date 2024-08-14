# pylint:disable = import-error
"""
This module defines methods for building a vector store.
"""
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_weaviate.vectorstores import WeaviateVectorStore
from loguru import logger
from src.models import SplitType, IndexConfig
from src.utils import semantic_split_sections, paragraph_split_sections

SEMANTIC_CHUNKING_MODEL = os.getenv(key='SEMANTIC_CHUNKING_MODEL',
                                    default="sentence-transformers/all-MiniLM-L6-v2")
DEVICE = os.getenv(key='DEVICE', default="cpu")
WEAVIATE_HOST = os.getenv(key='WEAVIATE_HOST', default="localhost")
WEAVIATE_PORT = os.getenv(key='WEAVIATE_PORT', default="8080")


class VectorStoreBuilder:
    """
    Handles the creation of a vector store from markdown files using a
    Weaviate client.
    """
    def __init__(self):
        # Load environment variables
        load_dotenv()

    @staticmethod
    def split_in_documents(md_file, index_config: IndexConfig):
        """
        Splits a markdown file into sections based on headers and optionally
        performs semantic chunking.
        :param: md_file: The markdown file content as a string.
        :param: index_config (IndexConfig): Configuration object specifying the
                embedding model and split type.
        :return: List: A list of documents.
        """
        hf_embedding = HuggingFaceEmbeddings(
            model_name=index_config.embedding_model_id,
            model_kwargs={'device': DEVICE},
            encode_kwargs={'normalize_embeddings': False}
        )

        headers_to_split_on = [("#", "Header_1"),
                               ("##", "Header_2"),
                               ("###", "Header_3"),
                               ("####", "Header_4")]
        text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        sections = text_splitter.split_text(md_file)

        if index_config.split_type == SplitType.SEMANTIC:
            return semantic_split_sections(sections, hf_embedding)

        return paragraph_split_sections(sections)

    @staticmethod
    def vector_store_from_documents(documents, weaviate_client, index_config):
        """
        Creates a Weaviate vector store from the provided documents.
        :param: documents (List): A list of documents representing
                the documents to be stored.
        :param: weaviate_client (weaviate.Client): An instance of the Weaviate client.
        :param: index_config (IndexConfig): Configuration object specifying the
                embedding model and index name.
        :return: tuple: A tuple containing the index name and the number of documents
                stored in the vector store.
        """
        hf_embedding = HuggingFaceEmbeddings(
            model_name=SEMANTIC_CHUNKING_MODEL,
            model_kwargs={'device': DEVICE},
            encode_kwargs={'normalize_embeddings': False}
        )
        index_name = index_config.index_name
        try:
            WeaviateVectorStore.from_documents(documents=documents,
                                               embedding=hf_embedding,
                                               client=weaviate_client,
                                               index_name=index_name)
            return index_name, len(documents)
        except Exception as e:
            logger.error("Error while creating the Weaviate Vector Store.")
            raise e
