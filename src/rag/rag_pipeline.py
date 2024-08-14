from langchain_core.runnables import RunnablePassthrough
from langchain.vectorstores.weaviate import Weaviate
from langchain import hub
from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser


class RAGPipeline:
    """

    """

    def __init__(self):
        pass

    @staticmethod
    def chat(retriever, model_id, message):
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
