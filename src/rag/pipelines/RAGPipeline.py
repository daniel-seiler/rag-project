import os
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

from haystack import Pipeline
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.routers import TextLanguageRouter
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.components.connectors.langfuse import LangfuseConnector

from rag.components.HypotheticalDocumentEmbedding import HypotheticalDocumentEmbedder

from typing import Tuple, List, Dict, Any

class RAGPipeline:
    """RAGPipeline is a class that sets up a Retrieval-Augmented Generation (RAG) pipeline for answering questions based on provided context and code snippets.
    Attributes:
        model (str): The model name for sentence transformers.
        document_store_url (str): The URL for the document store.
        rag_pipeline (Pipeline): The main pipeline object.
        template (str): The template for generating answers.
        llm_generator (OllamaChatGenerator): The language model generator.
        document_store (QdrantDocumentStore): The document store.
        language_router (TextLanguageRouter): The language router for handling different languages.
        prompt_embedder (SentenceTransformersTextEmbedder): The embedder for prompt text.
        document_retriever (QdrantEmbeddingRetriever): The retriever for fetching relevant documents.
        hyde_embedder (HypotheticalDocumentEmbedder): The embedder for hypothetical documents.
        prompt_builder (ChatPromptBuilder): The builder for chat prompts.
    Methods:
        __init__(model: str, document_store_url: str): Initializes the RAGPipeline with the specified model and document store URL.
        _build_pipeline() -> None: Builds the RAG pipeline by adding necessary components.
        _connect_components() -> None: Connects the components of the RAG pipeline.
        run(prompt: str, messages: List[Dict[str, Any]]) -> Tuple[bool, List[str]]: Runs the RAG pipeline with the given prompt and messages.
        _create_chat_messages(messages: List[Dict[str, Any]]) -> List[ChatMessage]: Creates chat messages from the provided list of messages."""
    
    def __init__(self, model:str="sentence-transformers/all-mpnet-base-v2", document_store_url:str="http://localhost:6333"):
        self.rag_pipeline = Pipeline()
        self.template = """
Answer the question based on the given context and code snippets, if code is provided.
Context:
{% for document in documents %} 
    {{ document.meta["full_content"] }}
    {% if document.meta["source"] != "" %}
        Link: https://pointclouds.org/documentation/{{ document.meta["source"] }}
    {% endif %}
{% endfor %}
Your answer should include links from the context that are relevant to the question.

Question: {{ prompt }}

Answer:
"""
        self.model = model
        self.document_store_url = document_store_url
        self.llm_generator = OllamaChatGenerator(model="llama3.1",
                            url="http://localhost:11434",
                            generation_kwargs={
                                "num_predict": -2,
                                "temperature": 0.1,
                            })
        self.document_store = QdrantDocumentStore(url=self.document_store_url)
        self.language_router = TextLanguageRouter(languages=["en"])
        self.prompt_embedder = SentenceTransformersTextEmbedder(model=self.model)
        self.document_retriever = QdrantEmbeddingRetriever(document_store=self.document_store)
        self.hyde_embedder = HypotheticalDocumentEmbedder(embedder_model=self.model)
        self.prompt_builder = ChatPromptBuilder(template=[ChatMessage.from_user(self.template)])
        self._build_pipeline()
        self._connect_components()

    def _build_pipeline(self) -> None:
        """
        Constructs the RAG (Retrieval-Augmented Generation) pipeline by adding various components to it.

        The following components are added to the pipeline:
        - "tracer": A LangfuseConnector for tracing the RAGPipeline.
        - "language_router": A component responsible for routing language-specific tasks.
        - "hyde_embedder": A component for embedding using the HYDE method.
        - "retriever": A document retriever component.
        - "prompt_builder": A component for building prompts.
        - "llm": A large language model generator component.

        Returns:
            None
        """
        self.rag_pipeline.add_component("tracer", LangfuseConnector("RAGPipeline trace"))
        self.rag_pipeline.add_component("language_router", self.language_router)
        self.rag_pipeline.add_component("hyde_embedder", self.hyde_embedder)
        self.rag_pipeline.add_component("retriever", self.document_retriever)
        self.rag_pipeline.add_component("prompt_builder", self.prompt_builder)
        self.rag_pipeline.add_component("llm", self.llm_generator)

    def _connect_components(self) -> None:
        """
        Connects the components of the RAG pipeline.

        This method establishes the connections between various components in the RAG pipeline:
        - Connects the English language router to the Hyde embedder.
        - Connects the hypothetical embedding from the Hyde embedder to the retriever's query embedding.
        - Connects the retriever to the prompt builder's documents.
        - Connects the prompt builder's prompt to the language model's messages.

        Returns:
            None
        """
        self.rag_pipeline.connect("language_router.en", "hyde_embedder.text") 
        self.rag_pipeline.connect("hyde_embedder.hypothetical_embedding", "retriever.query_embedding")
        self.rag_pipeline.connect("retriever", "prompt_builder.documents")
        self.rag_pipeline.connect("prompt_builder.prompt", "llm.messages")

    def run(self, prompt, messages) -> Tuple[bool, List[str]]:
        """
        Executes the RAG pipeline with the given prompt and messages.

        Args:
            prompt (str): The input prompt to be processed by the pipeline.
            messages (List[str]): A list of messages to be used in creating chat messages.

        Returns:
            Tuple[bool, List[str]]: A tuple where the first element is a boolean indicating
            whether the operation was successful, and the second element is a list of strings
            containing the response or an error message.
        """
        chat_messages = self._create_chat_messages(messages)
        output = self.rag_pipeline.run(data=
            {
                "language_router": {"text": prompt},
                "hyde_embedder": {"template": chat_messages},
                "retriever": {"top_k": 5},
                "prompt_builder": {"template_variables":{"prompt": prompt}, "template":chat_messages},
            }
        )
        if "language_router" in output.keys():
            return False, "Sorry, I can only answer questions in English."
        else:
            return True, output["llm"]["replies"][0].content
        
    def _create_chat_messages(self, messages:List[Dict[str, Any]]) -> List[ChatMessage]:
        """
        Converts a list of message dictionaries into a list of ChatMessage objects.

        Args:
            messages (List[Dict[str, Any]]): A list of dictionaries where each dictionary
                represents a message with keys "role" and "content". The "role" key should
                have values "user" or "assistant", and the "content" key should have the
                message content.

        Returns:
            List[ChatMessage]: A list of ChatMessage objects created from the input messages.
                The last message is replaced with a user message created from the template.
        """
        chat_messages = []
        for msg in messages:
            if msg["role"] == "user":
                chat_messages.append(ChatMessage.from_user(msg["content"]))
            else:
                chat_messages.append(ChatMessage.from_assistant(msg["content"]))
        chat_messages.pop()
        chat_messages.append(ChatMessage.from_user(self.template))
        return chat_messages