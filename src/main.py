# pylint:disable = import-error
"""
This module defines an API using FastAPI to interact with a Weaviate vector store.
It includes endpoints for creating a vector store from a markdown file and generating
chat responses. The API relies on dependencies for markdown cleaning, vector store building,
and a connection to a Weaviate instance.
"""
import os
import warnings
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Depends
from loguru import logger
import uvicorn

from data_processing import MarkdownCleaner
from src.models import IndexConfig, CreatedVectorStoreResponse
from src.utils import check_json
from vector_store import VectorStoreBuilder
from vector_store.vector_store_client import WeaviateClient


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI) -> None:
    """
    Manages the lifespan of the FastAPI application
    :param app: The FastAPI application instance.
    :return: None.
    """
    warnings.filterwarnings("ignore")

    logger.debug("Connecting to Weaviate Vector Store.")

    weaviate_host = os.getenv(key='WEAVIATE_HOST', default="localhost")
    weaviate_port = os.getenv(key='WEAVIATE_PORT', default="8080")

    weaviate_client = WeaviateClient(weaviate_host, weaviate_port).client
    fastapi_app.state.weaviate_client = weaviate_client
    logger.debug("RAG Pipeline is up!")

    yield


app = FastAPI(lifespan=lifespan)


#@app.post(
#    "/v1/chat/completion",
#    operation_id="chat_completion",
#    summary="Creates a response for the given converstion."
#)
#async def chat_completion(request):
#    pass


@app.post(
    "/v1/vector_store/docs/",
    operation_id="create_vs_docs_from_markdown",
    summary="Creates a vector store from a markdown file."
)
def create_vs_docs_from_markdown(
        index_config: IndexConfig = Depends(check_json),
        md_file: UploadFile = File(...),
):
    """
    Endpoint to create a vector store from an uploaded markdown file.
    The uploaded markdown file is cleaned, split into documents, and then stored in a
    Weaviate vector store based on the provided index configuration.
    :param: index_config: Configuration of index in the vector store
    :param: md_file (UploadFile): The markdown file uploaded by the user
    :return: CreatedVectorStoreResponse: The response containing index name and number of documents.
    """
    clean_md = MarkdownCleaner.markdown_processing(md_file)
    documents = VectorStoreBuilder.split_in_documents(clean_md, index_config)
    index_name, num_docs = VectorStoreBuilder.vector_store_from_documents(documents,
                                                                          app.state.weaviate_client,
                                                                          index_config)

    return CreatedVectorStoreResponse(index_name=index_name, inserted_count=num_docs)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
