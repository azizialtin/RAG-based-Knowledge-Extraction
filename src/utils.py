# pylint:disable = import-error
"""
This module defines utility functions and methods.
"""
from typing import List
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi import HTTPException, Form
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError
from starlette import status
from loguru import logger
from src.models import IndexConfig


def check_json(index_config: str = Form(...)):
    """
    Validates a JSON string against the IndexConfig model.
    This function attempts to validate the provided JSON string by
    converting it to an
    IndexConfig object. If validation fails, it logs the exception
    and raises an HTTPException.
    :param: index_config (str): The JSON string representing the
    index configuration.
    :return: IndexConfig: The validated IndexConfig object.
    """

    try:
        return IndexConfig.model_validate_json(index_config)
    except ValidationError as e:
        logger.exception("Validation failed for the provided JSON string.")
        raise HTTPException(
            detail=jsonable_encoder(e.errors()),
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
        ) from e


def semantic_split_sections(sections: List[Document], hf_embedding):
    """
    Splits a list of documents into smaller chunks using semantic chunking.
    This function uses a HuggingFace embedding model to perform semantic
    chunking on documents that exceed a certain length. Each chunk is
    treated as a new document.
    :param: sections (List[Document]): The list of documents to be split.
    :param: hf_embedding (HuggingFaceEmbeddings): The embedding model
    used for semantic chunking.
    :return: List[Document]: A list of documents,
    where long documents are split into semantically meaningful chunks.
    """
    documents = []
    for section in sections:
        # If the documents are too long we split the documents by using
        # semantic chunking
        if len(section.page_content.split()) > 400:
            semantic_chunker = SemanticChunker(hf_embedding)
            chunks = semantic_chunker.split_text(section.page_content)
            for chunk in chunks:
                # Create a new Document for each chunk with metadata
                chunk_document = Document(
                    page_content=chunk,
                    metadata=section.metadata
                )
                documents.append(chunk_document)
        else:
            documents.append(section)
    return documents


def recursive_split_sections(sections: List[Document]):
    """
    Splits a list of document sections into smaller chunks using a recursive
    character splitter.
    :param sections: (List[Document]): A list of Document objects to be split.
    :return: List[Document]: A list of Document objects where each section has
    been split into smaller chunks.
    """
    chunk_size = 1900
    chunk_overlap = 100

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n\n", "\n\n", "\n", "."]
    )

    splits = text_splitter.split_documents(sections)
    return splits


def load_text_file(file_path: str) -> str:
    """
    Reads a txt file and returns a string.
    :param file_path: path of the file
    :return: string
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
