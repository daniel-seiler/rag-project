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

from components.HypotheticalDocumentEmbedding import HypotheticalDocumentEmbedder

from typing import Tuple, List

class RAGPipeline:
    def __init__(self, model="sentence-transformers/all-mpnet-base-v2", document_store_url="http://localhost:6333"):
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

    def _build_pipeline(self):
        self.rag_pipeline.add_component("tracer", LangfuseConnector("RAGPipeline trace"))
        self.rag_pipeline.add_component("language_router", self.language_router)
        # self.rag_pipeline.add_component("embedder", self.prompt_embedder)
        self.rag_pipeline.add_component("hyde_embedder", self.hyde_embedder)
        self.rag_pipeline.add_component("retriever", self.document_retriever)
        self.rag_pipeline.add_component("prompt_builder", self.prompt_builder)
        self.rag_pipeline.add_component("llm", self.llm_generator)

    def _connect_components(self):
        self.rag_pipeline.connect("language_router.en", "hyde_embedder.text") 
        self.rag_pipeline.connect("hyde_embedder.hypothetical_embedding", "retriever.query_embedding")
        self.rag_pipeline.connect("retriever", "prompt_builder.documents")
        self.rag_pipeline.connect("prompt_builder.prompt", "llm.messages")

    def run(self, prompt, messages) -> Tuple[bool, List[str]]:
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
        
    def _create_chat_messages(self, messages):
        chat_messages = []
        for msg in messages:
            if msg["role"] == "user":
                chat_messages.append(ChatMessage.from_user(msg["content"]))
            else:
                chat_messages.append(ChatMessage.from_assistant(msg["content"]))
        chat_messages.pop()
        chat_messages.append(ChatMessage.from_user(self.template))
        return chat_messages