from haystack import Pipeline
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever


class RAGPipeline:
    def __init__(self, model="sentence-transformers/all-mpnet-base-v2", document_store_url="http://localhost:6333"):
        self.rag_pipeline = Pipeline()
        self.template = """
Beantworte die Frage basierend des gegebenen Kontexts.

Kontext:
{% for document in documents %} 
    {{ document.meta["original_text"] }}
{% endfor %}

Frage: {{ question }}
Antwort:
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
        self.prompt_embedder = SentenceTransformersTextEmbedder(model=self.model)
        self.document_retriever = QdrantEmbeddingRetriever(document_store=self.document_store)
        self.prompt_builder = ChatPromptBuilder(template=[ChatMessage.from_user(self.template)])
        self._build_pipeline()
        self._connect_components()

    def _build_pipeline(self):
        self.rag_pipeline.add_component("embedder", self.prompt_embedder)
        self.rag_pipeline.add_component("retriever", self.document_retriever)
        self.rag_pipeline.add_component("prompt_builder", self.prompt_builder)
        self.rag_pipeline.add_component("llm", self.llm_generator)

    def _connect_components(self):
        self.rag_pipeline.connect("embedder.embedding", "retriever.query_embedding")
        self.rag_pipeline.connect("retriever", "prompt_builder.documents")
        self.rag_pipeline.connect("prompt_builder.prompt", "llm.messages")

    def run(self, prompt, messages):
        chat_messages = self._create_chat_messages(messages)
        return self.rag_pipeline.run(data=
            {
                "embedder": {"text": prompt},
                "prompt_builder": {"template_variables":{"question": prompt}, "template":chat_messages},
            }
        )['llm']
        
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