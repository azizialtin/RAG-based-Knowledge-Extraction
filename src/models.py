# pylint:disable=line-too-long
"""
This module defines data models and enumerations for configuring and
interacting with a vector store.
"""

from enum import Enum
from pydantic import BaseModel


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
    embedding_model_id: str
    split_type: SplitType


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
    model_id: str


class RetrievalConfigs(BaseModel):
    """
    Model representing the configuration for retrieval tasks.
    Attributes:
        model_id (str): The identifier for the model used to generate
        embeddings for retrieval.
        top_k (int): The number of top documents to retrieve based on
        similarity.
    """
    model_id: str
    top_k: int


class ChatConfigs(BaseModel):
    """
    Model representing the overall configuration for a chat session.
    Attributes:
        index_name (str): The name of the index to use for retrieval.
        generation_configs (GenerationConfigs): Configuration settings
        for the response generation model.
        retrieval_configs (RetrievalConfigs): Configuration settings
        for the retrieval model.
    """
    index_name: str
    generation_configs: GenerationConfigs
    retrieval_configs: RetrievalConfigs


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
