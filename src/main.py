# pylint:disable = import-error
"""
This module defines an API using FastAPI to interact with a vector store.
It includes endpoints for creating a vector store from a markdown file
and generating chat responses.
"""
import warnings
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from loguru import logger
import uvicorn
from data_processing import MarkdownCleaner

from src.models import (IndexConfig,
                        CreatedVectorStoreResponse,
                        CompletionRequest, CompletionResponse)
from src.rag.rag_pipeline import RAGPipeline
from src.utils import check_json
from vector_store import VectorStoreBuilder


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI) -> None:
    """
    Manages the lifespan of the FastAPI application.
    :param fastapi_app: The FastAPI application instance.
    :return: None.
    """
    warnings.filterwarnings("ignore")
    logger.info("RAG Pipeline is up!")

    # Initialize shared resources
    fastapi_app.state.memory = ConversationBufferMemory(
        return_messages=True, output_key="answer", input_key="question"
    )

    yield

    # Clean up resources if necessary
    logger.info("RAG Pipeline is shutting down.")


app = FastAPI(
    title="RAG API",
    description="This API allows you to create vector stores from markdown files and generate "
                "chat responses using a RAG (Retrieval-Augmented Generation) pipeline.",
    version="0.1.0",
    lifespan=lifespan
)


@app.post(
    "/v1/chat/completion/",
    operation_id="chat_completion",
    summary="Generate a chat response",
    response_model=CompletionResponse,
    response_description="The generated chat response",
    tags=["Chat Completion"]
)
async def chat_completion(request: CompletionRequest):
    """
    Generates a chat response based on the provided conversation and
    configuration.
    :param request: CompletionRequest object containing conversation
                    details and configurations.
    :return: The generated answer from the RAG pipeline.
    """
    try:
        top_k = request.configs.retrieval_configs.top_k
        embedding_model = request.configs.retrieval_configs.model_id

        logger.debug("Loading the Embedding model from Hugging Face.")
        hf_embedding = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': "cpu"},
            encode_kwargs={'normalize_embeddings': False},
            show_progress=True
        )

        logger.debug("Loading the persisted Vector Store.")
        collection = Chroma(
            collection_name=request.configs.index_name,
            persist_directory="../data/docs/",
            embedding_function=hf_embedding
        )

        logger.debug("Initializing the Retriever.")
        retriever = collection.as_retriever(search_kwargs={"k": top_k})

        pipeline = RAGPipeline()
        logger.debug("Initializing the Reranker.")
        pipeline.initialize_reranker(
            base_retriever=retriever,
            reranker_configs=request.configs.reranking_configs
        )

        logger.debug("Initializing the LLM.")
        pipeline.initialize_llm(
            request.configs.generation_configs
        )

        logger.debug("Creating the Chat Chain.")
        chat = pipeline.get_chat_chain(app.state.memory)

        logger.debug("Generating the Chat Response:")
        result = chat(request.message)
        print()
        return CompletionResponse(
            answer=result["answer"],
            relevant_documents=str(result["documents"])
        )

    except Exception as e:
        logger.error(f"Error during chat completion: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error"
        ) from e


@app.post(
    "/v1/vector_store/docs/",
    operation_id="create_vs_docs_from_markdown",
    summary="Create a vector store from a markdown file",
    response_model=CreatedVectorStoreResponse,
    response_description="Details of the created vector store",
    tags=["Vector Store Creation"]
)
def create_vs_docs_from_markdown(
        index_config: IndexConfig = Depends(check_json),
        md_file: UploadFile = File(...),
):
    """
    Endpoint to create a vector store from an uploaded markdown file
    :param: index_config: Configuration of index in the vector store
    :param: md_file (UploadFile): The markdown file uploaded by the user
    :return: CreatedVectorStoreResponse: The response containing index
    name and number of documents.
    """
    try:
        logger.debug("Cleaning the Markdown File...")
        clean_md = MarkdownCleaner.markdown_processing(md_file)

        logger.debug("Splitting the Markdown File into documents...")
        vs = VectorStoreBuilder()
        documents = vs.split_in_documents(clean_md, index_config)

        logger.debug("Building the vector store...")
        index_name, num_docs = vs.vector_store_from_documents(
            documents=documents,
            index_config=index_config
        )

        return CreatedVectorStoreResponse(
            index_name=index_name,
            inserted_count=num_docs
        )

    except Exception as e:
        logger.error(f"Error while creating the vector store: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error"
        ) from e


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
