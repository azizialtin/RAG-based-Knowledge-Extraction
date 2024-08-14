import os
import json
import time
import weaviate
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_core.documents.base import Document
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_community.vectorstores import Weaviate

from loguru import logger

SEMANTIC_CHUNKING_MODEL = os.getenv(key='SEMANTIC_CHUNKING_MODEL', default="sentence-transformers/all-mpnet-base-v2")
DEVICE = os.getenv(key='DEVICE', default="cpu")
WEAVIATE_HOST = os.getenv(key='WEAVIATE_HOST', default="localhost")
WEAVIATE_PORT = os.getenv(key='WEAVIATE_PORT', default=8080)

class VectorStoreBuilder:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        while not self.check_weaviate_server(host=WEAVIATE_HOST, port=WEAVIATE_PORT):
            logger.debug("Waiting for the Weaviate server to start...")
            time.sleep(5)
        logger.info("Weaviate Server is up!")

        global weaviate_client
        weaviate_client = weaviate.connect_to_local(host="localhost", port=8080)

    @staticmethod
    def check_weaviate_server(
            host: str,
            port: int
    ) -> bool:
        try:
            weaviate.connect_to_local(host=host, port=port)
            return True
        except Exception:
            return False

    @staticmethod
    def split_in_documents(md_file):

        hf_embedding = HuggingFaceEmbeddings(
            model_name=SEMANTIC_CHUNKING_MODEL,
            model_kwargs={'device': DEVICE},
            encode_kwargs={'normalize_embeddings': False}
        )

        headers_to_split_on = [("#", "Header_1"), ("##", "Header_2"), ("###", "Header_3"), ("####", "Header_4")]
        text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        sections = text_splitter.split_text(md_file)
        documents = []
        for section in sections:
            # If the documents are too long we split the documents by using semantic chunking
            if len(section.page_content.split()) > 400:
                semantic_chunker = SemanticChunker(hf_embedding)
                chunks = semantic_chunker.split_text(section.page_content)
                for chunk in chunks:
                    # Create a new Document for each chunk with appropriate metadata
                    chunk_document = Document(
                        page_content=chunk,
                        metadata=section.metadata
                    )
                    documents.append(chunk_document)
            else:
                documents.append(section)
        return documents

    @staticmethod
    def vector_store_from_documents(documents):
        hf_embedding = HuggingFaceEmbeddings(
            model_name=SEMANTIC_CHUNKING_MODEL,
            model_kwargs={'device': DEVICE},
            encode_kwargs={'normalize_embeddings': False}
        )
        try:
            vector_store = WeaviateVectorStore.from_documents(documents, hf_embedding, client=weaviate_client, index_name="altin")
            return vector_store
        except Exception as e:
            logger.error(f"Error while creating the Weaviate vector store.")
            raise e