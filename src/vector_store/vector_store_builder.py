# pylint:disable = import-error
"""
This module defines methods for building a vector store.
"""
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import MarkdownHeaderTextSplitter
from loguru import logger

from src.models import SplitType, IndexConfig
from src.utils import semantic_split_sections, paragraph_split_sections

# Load environment variables at the beginning of the script
load_dotenv()
DEVICE = os.getenv(key='DEVICE', default="cpu")


class VectorStoreBuilder:
    """
    Handles the creation of a vector store from markdown files using a
    Vector Store client.
    """
    HEADERS_TO_SPLIT_ON = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
        ("####", "Header_4")
    ]

    def __init__(self):
        """
        Initializes an instance of the VectorStoreBuilder class.
        """

    @staticmethod
    def _create_embedding(
            embedding_model_id: str
    ) -> HuggingFaceEmbeddings:
        """
        Creates a HuggingFace embedding model based on the given
        index configuration.
        :param embedding_model_id: embedding model id.
        :return: HuggingFaceEmbeddings object.
        """
        return HuggingFaceEmbeddings(
            model_name=embedding_model_id,
            model_kwargs={'device': DEVICE},
            encode_kwargs={'normalize_embeddings': False}
        )

    @staticmethod
    def split_in_documents(md_file: str, index_config: IndexConfig) -> list:
        """
        Splits a markdown file into sections based on headers and optionally
        performs semantic chunking.
        :param: md_file: The markdown file content as a string.
        :param: index_config (IndexConfig): Configuration object specifying the
                embedding model and split type.
        :return: List: A list of documents.
        """
        try:
            hf_embedding = VectorStoreBuilder._create_embedding(
                embedding_model_id=index_config.embedding_model_id
            )
        except Exception as e:
            logger.error("Error while creating embedding model, "
                         "make sure the provided name is correct.")
            raise e

        text_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=VectorStoreBuilder.HEADERS_TO_SPLIT_ON
        )
        sections = text_splitter.split_text(md_file)

        if index_config.split_type == SplitType.SEMANTIC:
            return semantic_split_sections(sections, hf_embedding)

        return paragraph_split_sections(sections)

    @staticmethod
    def vector_store_from_documents(
            documents: list,
            index_config: IndexConfig
    ):
        """
        Creates a vector store from the provided documents.
        :param: documents (List): A list of documents representing
                the documents to be stored.
        :param: index_config (IndexConfig): Configuration object specifying the
                embedding model and index name.
        :return: tuple: A tuple containing the index name and the
                number of documents stored in the vector store.
        """
        hf_embedding = HuggingFaceEmbeddings(
            model_name=index_config.embedding_model_id,
            model_kwargs= {'device': DEVICE},
            encode_kwargs={'normalize_embeddings': False}
        )
        index_name = index_config.index_name

        try:
            chroma_vs = Chroma.from_documents(
                documents=documents,
                embedding=hf_embedding,
                persist_directory="../data/docs",
                collection_name=index_name
            )
            chroma_vs.persist()
            return index_name, len(documents)

        except Exception as e:
            logger.error(f"Error while creating the Vector Store. {e}")
            raise e
