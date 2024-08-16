# RAG-based-Knowledge-Extraction

This project provides a RESTful API built using FastAPI to interact with a Retrieval-Augmented Generation (RAG) pipeline.
The API allows users to create a vector store from Markdown files and generate chat responses using an integrated conversational model.

## Prerequisites

* Python = 3.10
* Ollama version 0.1.26 or higher.

## Setup

1. **Clone the Repository:** Clone this repository to your local machine.
2. **Create a Python virtual environment**: run `running python3 -m venv .venv`
3. **Activate the virtual environment**: run source `.venv/bin/activate` on Unix or MacOS, 
or `.\.venv\Scripts\activate` on Windows
4. **Install Dependencies:** run `pip install -r requirements.txt`

## Running the Project
1. Ensure your virtual environment is activated
2. Run the main script with `python main.py`

### Usage
The project offers two primary functionalities:

* Vector Store Creation: Upload Markdown files and automatically 
create a vector store for efficient information retrieval.
* Chat Completion: Generate context-aware chat responses using a 
RAG pipeline that integrates document retrieval with conversational AI.

#### Creating a Vector Store from Markdown
To create a vector store from a Markdown file, send a POST request to the `/v1/vector_store/docs/`endpoint with the Markdown file attached.
This saves the vector store locally, and it can be loaded later during generation.


```
POST /v1/vector_store/docs/
```
**Description:** Creates a vector store from an uploaded Markdown file.

**Request:**
`index_config`: JSON configuration for the vector store index.
`md_file`: The Markdown file to process and store.

**Response:**
`index_name`: The name of the created index.
`inserted_count`: Number of documents inserted into the vector store.

**Example in postman:**
![img.png](data%2Fimages_readme%2Fimg.png)

**Sample** `index_config`:
```
{
    "index_name": "Git_tutorial",
    "embedding_model_id": "sentence-transformers/all-MiniLM-L6-v2",
    "split_type": "paragraph"
}
```

* `index_name`: represents the name of the index in the vector store
* `embedding_model_id`: Name of the embedding model. You can use any sentence-transformers model from HuggingFace.
https://huggingface.co/models?library=sentence-transformers
* `split_type`:  Method for splitting long documents 
("semantic" for semantic splitting or "paragraph" for paragraph splitting).


#### Chat Completion
To generate a chat response, send a POST request to the /v1/chat/completion/ endpoint with the 
conversation history and configuration details.
```
POST /v1/chat/completion/
```

**Description:** Generates a response for a given message using the RAG pipeline.

**Request:**
`message`: The input message or conversation history.
`configs`: Configuration for retrieval and generation (e.g., models, top_k).

**Response:**
`answer`: The generated chat response.

**Example in Postman**
![img_1.png](data%2Fimages_readme%2Fimg_1.png)

## RAG Pipeline Explained
The Retrieval-Augmented Generation (RAG) pipeline enhances response generation by integrating document retrieval with conversational AI. The pipeline includes the following steps:
* **Data cleaning:**  Cleans the input Markdown files.
* **Data splitting:** Splits the Markdown content based on header levels. 
If a chunk is too large, it splits further by paragraph or using semantic splitting.

**Pipeline Steps::**
* **Standalone question:** Transforms the user's message into a standalone question.
* **Retrieve Documents:** Retrieves relevant documents based on the question.
* **Input Construction:** Combines the retrieved documents with the original question to generate the final response.


## Libraries Used 
* **Langchain:** A Python library for working with Large Language Model
* **Ollama:** A platform for running Large Language models locally.
* **Chroma:** A vector database for storing and retrieving embeddings.