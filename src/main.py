from data_processing import Markdown_Cleaner
from vector_store import VectorStoreBuilder
from fastapi import FastAPI, File, UploadFile
from loguru import logger
import os
import weaviate
import time
import uvicorn


app = FastAPI()


def check_weaviate_server(
        host: str,
        port: int
) -> bool:
    try:
        weaviate.connect_to_local(host=host, port=port)
        return True
    except Exception:
        return False


@app.on_event("startup")
async def startup_event() -> None:
    logger.debug("Connecting to Weaviate Vector Store.")

    WEAVIATE_HOST = os.getenv(key='WEAVIATE_HOST', default="localhost")
    WEAVIATE_PORT = os.getenv(key='WEAVIATE_PORT', default=8080)

    while not check_weaviate_server(host=WEAVIATE_HOST, port=WEAVIATE_PORT):
        logger.debug("Waiting for the Weaviate server to start...")
        time.sleep(5)
    logger.info("Weaviate Server is up!")

    global weaviate_client
    weaviate_client = weaviate.connect_to_local(host=WEAVIATE_HOST, port=WEAVIATE_PORT)

    logger.debug("RAG Pipeline is up!")


@app.post(
    "/v1/chat/completion",
    operation_id="chat_completion",
    #response_model=CompletionResponse,
    summary="Creates a response for the given converstion."
)
async def chat_completion(CompletionRequest):
    pass



@app.post(
    "/v1/vector_store/docs/{index_name}",
    operation_id="chat_completion",
    # response_model=CompletionResponse,
    summary="Creates a response for the given converstion."
)
def create_vs_docs_from_markdown(
        index_name: str,
        md_file: UploadFile = File(...),
    ):

    clean_md = Markdown_Cleaner.markdown_processing(md_file)
    documents = VectorStoreBuilder.split_in_documents(md_file=clean_md)
    vector_store = VectorStoreBuilder.vector_store_from_documents(documents)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
