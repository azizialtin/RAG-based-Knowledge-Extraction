# pylint:disable = line-too-long
"""
This module defines data models and enumerations for configuring and interacting with a vector store.
"""

from enum import Enum
from pydantic import BaseModel


class SplitType(str, Enum):
    """
    Enumeration representing the methods for splitting a document into smaller segments.
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
        index_name (str): The name of the index to be created in the vector store.
        embedding_model_id (str): The identifier for the embedding model used to generate vector representations.
        split_type (SplitType): The method used to split the document before indexing.
    """

    index_name: str
    embedding_model_id: str
    split_type: SplitType


class CreatedVectorStoreResponse(BaseModel):
    """
    Model representing the response after successfully creating a vector store.
    Attributes:
        index_name (str): The name of the index.
        inserted_count (int): The number of documents.
    """
    index_name: str
    inserted_count: int
