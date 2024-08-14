# pylint:disable = import-error, too-few-public-methods
"""
This module defines all methods needed for the RAG pipeline
"""

from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline


class RAGPipeline:
    """
    Class that encapsulates the RAG pipeline, providing methods for interacting with
    a retriever and a language model to generate contextually aware responses.
    """

    def __init__(self):
        """
        Initializes an instance of the RAGPipeline class.
        """

    @staticmethod
    def chat(retriever, model_id, message):
        """
        Generates a response by combining document retrieval and language model generation.
        This method first retrieves relevant documents using the provided retriever and then
        uses a HuggingFace language model to generate a response based on the retrieved documents
        and the given message.
        :param retriever: The document retriever used to fetch relevant documents.
        :param model_id: The identifier for the HuggingFace model used for text generation
        :param message: The input message or question that the pipeline will respond to.
        :return:
        """
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)


        llm = HuggingFacePipeline.from_model_id(
            model_id=model_id,
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": 100,
                "top_k": 50,
                "temperature": 0.1,
            },
        )

        prompt = hub.pull("rlm/rag-prompt")

        rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

        rag_chain.invoke(message)
