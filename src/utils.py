# pylint:disable = import-error
"""
This module defines utility functions and methods.
"""
from typing import List
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
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


def paragraph_split_sections(sections: List[Document]):
    """
    Splits a list of documents into smaller sections based on paragraphs.
    This function splits the content of each document by paragraphs,
    keeping related
    lines together, such as bullet points. Each paragraph is treated
    as a new document.
    :param: sections (List[Document]): The list of documents to be split.
    :return: List[Document]: A list of documents, where each paragraph
    is a separate document.
    """
    documents = []
    for section in sections:
        content = section.page_content
        paragraphs = []
        current_paragraph = []

        for line in content.splitlines():
            line = line.strip()
            # Bullet points start a new paragraph
            if line.startswith(("-", "*")):
                if current_paragraph:
                    paragraphs.append(" ".join(current_paragraph))
                    current_paragraph = []
                current_paragraph.append(line)
            elif line:
                current_paragraph.append(line)
            else:
                if current_paragraph:
                    paragraphs.append(" ".join(current_paragraph))
                    current_paragraph = []

        # Add the last paragraph if there's any content left
        if current_paragraph:
            paragraphs.append(" ".join(current_paragraph))

        # Create a new Document for each paragraph
        for paragraph in paragraphs:
            documents.append(
                Document(
                    page_content=paragraph,
                    metadata=section.metadata)
            )

    return documents
