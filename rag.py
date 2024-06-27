# This file is responsible for creating embeddings and writing to the database for Retrieval
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

class Chat:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model="phi3:3.8b")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
        self.prompt = PromptTemplate.from_template(
    """
    <s>[INST] You are an AI language model assistant specializing in Cyber Security Acts. Your task is to:

    1. Analyze the user's question from multiple perspectives to generate a comprehensive response.
    2. Retrieve relevant information from a vector database to support your answer.
    3. Overcome limitations of distance-based similarity search by considering various interpretations of the query.

    Guidelines:
    - Provide detailed information only when the user's question clearly requires it.
    - For simple greetings or vague questions, respond concisely and appropriately without retrieving detailed information.
    - Use Markdown formatting in your responses for better readability.
    - Include relevant snippets from the provided context to support your answer.

    [/INST]</s>

    [INST] Question: {question}
    Context: {context}
    Answer: [/INST]
    """
)

    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)
        self.vector_store = Chroma.from_documents(documents=chunks, embedding=self.embeddings)
        base_retriever = self.vector_store.as_retriever(
            search_type="mmr",  # Use Maximum Marginal Relevance
            search_kwargs={
                "k": 5,
                "fetch_k": 20,  
            },
        )
        # Implement contextual compression for better retrieval
        compressor = LLMChainExtractor.from_llm(self.model)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."
        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
