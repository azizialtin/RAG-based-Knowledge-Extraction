# pylint:disable=line-too-long
"""
This module defines data models and enumerations for configuring and
interacting with a vector store.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict


class SplitType(str, Enum):
    """
    Enumeration representing the methods for splitting a document
    into smaller segments.
    Attributes:
        SEMANTIC (str): Split the document based on semantic understanding.
        PARAGRAPH (str): Split the document by paragraphs.
    """
    SEMANTIC = "semantic"
    PARAGRAPH = "paragraph"


class IndexConfig(BaseModel):
    """
    Model representing the configuration for creating a vector store index.
    Attributes:
        index_name (str): The name of the index to be created in the
        vector store.
        embedding_model_id (str): The identifier for the embedding model
        used to generate vector representations.
        split_type (SplitType): The method used to split the document
        before indexing.
    """
    index_name: str
    embedding_model_id: Optional[str] = "sentence-transformers/all-MiniLM-L12-v2"
    split_type: Optional[SplitType] = SplitType.PARAGRAPH


class CreatedVectorStoreResponse(BaseModel):
    """
    Model representing the response after successfully creating a
    vector store.
    Attributes:
        index_name (str): The name of the index.
        inserted_count (int): The number of documents inserted into
        the vector store.
    """
    index_name: str
    inserted_count: int


class GenerationConfigs(BaseModel):
    """
    Model representing the configuration for generation tasks.
    Attributes:
        model_id (str): The identifier for the model used to generate
        responses.
    """
    model_id: Optional[str] = "gemma2:2b"


class RetrievalConfigs(BaseModel):
    """
    Model representing the configuration for retrieval tasks.
    Attributes:
        model_id (str): The identifier for the model used to generate
        embeddings for retrieval.
        top_k (int): The number of top documents to retrieve based on
        similarity.
    """
    model_id: Optional[str] = "sentence-transformers/all-MiniLM-L12-v2"
    top_k: Optional[int] = 10


class RerankingConfigs(BaseModel):
    """
    Model representing the configuration for reranking tasks.
    Attributes:
        model_id (str): The identifier for the model used to generate
        embeddings for reranking.
        top_k (int): The number of top documents to compress to
        after reranking.
    """
    model_id: Optional[str] = "ms-marco-MiniLM-L-12-v2"
    top_k: Optional[int] = 5


class ChatConfigs(BaseModel):
    """
    Model representing the overall configuration for a chat session.
    Attributes:
        index_name (str): The name of the index to use for retrieval.
        retrieval_configs (RetrievalConfigs): Configuration settings
        for the retrieval model.
        reranking_cofigs (RerankingConfigs): Configuration settings
        for the reranker model.
        generation_configs (GenerationConfigs): Configuration settings
        for the response generation model.
    """
    index_name: str
    retrieval_configs: RetrievalConfigs = RetrievalConfigs()
    reranking_configs: RerankingConfigs = RerankingConfigs()
    generation_configs: GenerationConfigs = GenerationConfigs()
    model_config = ConfigDict(arbitrary_types_allowed=True)


class CompletionRequest(BaseModel):
    """
    Model representing a request for chat completion.
    Attributes:
        message (str): The input message to be processed.
        configs (ChatConfigs): Configuration settings for processing
        the message.
    """
    message: str
    configs: ChatConfigs


class CompletionResponse(BaseModel):
    """
    Model representing a response for chat completion.
    Attributes:
        answer (str): The input message to be processed.
        relevant_documents (str): Retrieved documents from the retriever
    """
    answer: str
    relevant_documents: str
