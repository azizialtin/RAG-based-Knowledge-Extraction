# pylint:disable = import-error, too-few-public-methods, line-too-long, no-name-in-module, fixme
"""
This module defines all methods needed for the RAG pipeline
"""
from operator import itemgetter

from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_community.llms.ollama import Ollama
from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import format_document, ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from loguru import logger
from src.models import RerankingConfigs, GenerationConfigs
from src.rag.language_models import is_model_available
from src.utils import load_text_file

# Load the text file
ANSWER_GENERATION_PATH = './rag/prompt_templates/answer_generation_prompt.txt'
CONDENSE_QUESTION_PATH = './rag/prompt_templates/condense_question_prompt.txt'
DEFAULT_DOCUMENT_PATH = './rag/prompt_templates/default_document_prompt.txt'

# Load the templates
ANSWER_PROMPT = ChatPromptTemplate.from_template(load_text_file(ANSWER_GENERATION_PATH))
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(load_text_file(CONDENSE_QUESTION_PATH))
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(load_text_file(DEFAULT_DOCUMENT_PATH))


class RAGPipeline:
    """
    Class that encapsulates the RAG pipeline, providing methods
    for interacting with a retriever and a language model to
    generate contextually aware responses.
    """

    def __init__(self):
        """
        Initializes an instance of the RAGPipeline class.
        """
        # Set up the reranked retriever
        self.compression_retriever = None
        # Set up the llm
        self.llm = None

    def initialize_reranker(
            self,
            base_retriever: VectorStoreRetriever,
            reranker_configs: RerankingConfigs
    ):
        """
        Initializes the reranker
        :param base_retriever: An instance of `VectorStoreRetriever` that serves
        as the base retriever for the reranker.
        :param reranker_configs:  An instance of `RerankingConfigs` containing the
        configuration settings for the reranker, including the model ID and the
        number of top results to retain.
        """
        compressor = FlashrankRerank(
            model=reranker_configs.model_id,
            top_n=reranker_configs.top_k
        )
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )

    def initialize_llm(self, gen_configs: GenerationConfigs):
        """
        Initializes the large language model (LLM) for text generation.
        :param gen_configs: An instance of `GenerationConfigs` containing the
        configuration settings for the language model, including the model ID to
        be used.
        """
        llm = Ollama(model=gen_configs.model_id)

        logger.debug("Checking if the LLM is available.")
        is_model_available(gen_configs.model_id)

        self.llm = llm

    @staticmethod
    def _combine_documents(
            documents: list,
            document_prompt: PromptTemplate = DEFAULT_DOCUMENT_PROMPT,
            document_separator: str = "\n\n"
    ) -> str:
        """
        Combines a list of documents into a single string, formatted
        according to the provided prompt.
        :param documents: List of documents to combine.
        :param document_prompt: The prompt template to use for
        formatting each document.
        :param document_separator: The separator to use between
        documents.
        :return: A single string containing all combined and
        formatted documents.
        """
        doc_strings = [format_document(document, document_prompt) for document in documents]
        return document_separator.join(doc_strings)

    def get_chat_chain(
            self,
            memory: ConversationBufferMemory
    ):
        """
        Constructs the RAG pipeline's chat chain.
        :param memory: The memory object that maintains chat history.
        :return: A function that takes a question as input and returns
        the answer along with retrieved documents.
        """

        def load_memory_chain():
            return RunnablePassthrough.assign(
                chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
            )

        def create_standalone_question_chain():
            return {
                "standalone_question": {
                    "question": lambda x: x["question"],
                    "chat_history": lambda x: get_buffer_string(x["chat_history"]),
                }
                | CONDENSE_QUESTION_PROMPT
                | self.llm
            }

        def retrieve_documents_chain():
            return {
                "documents": itemgetter("standalone_question") | self.compression_retriever,
                "question": lambda x: x["standalone_question"],
            }

        def create_final_inputs_chain():
            return {
                "context": lambda x: RAGPipeline._combine_documents(x["documents"]),
                "question": itemgetter("question"),
            }

        def create_answers_chain():
            return {
                "answer": create_final_inputs_chain()
                | ANSWER_PROMPT
                | self.llm.with_config(callbacks=[StreamingStdOutCallbackHandler()]),
                "documents": itemgetter("documents"),
            }

        final_chain = (
            load_memory_chain()
            | create_standalone_question_chain()
            | retrieve_documents_chain()
            | create_answers_chain()
        )

        def chat(question: str):
            inputs = {"question": question}
            result = final_chain.invoke(inputs)
            memory.save_context(inputs, {"answer": result["answer"]})
            return result

        return chat
