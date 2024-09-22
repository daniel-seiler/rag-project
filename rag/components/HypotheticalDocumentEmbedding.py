from haystack import Pipeline, component, Document, default_to_dict, default_from_dict
from haystack.dataclasses import ChatMessage
from haystack.components.converters import OutputAdapter
from haystack.components.embedders.sentence_transformers_document_embedder import SentenceTransformersDocumentEmbedder
from haystack.components.builders import ChatPromptBuilder

from haystack_integrations.components.generators.ollama import OllamaChatGenerator

from typing import Dict, Any, List, Optional
from numpy import array, mean
from haystack.utils import Secret


@component
class HypotheticalDocumentEmbedder:

    def __init__(
        self,
        instruct_llm: str = "llama3.1",
        nr_completions: int = 5,
        embedder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.instruct_llm = instruct_llm
        self.nr_completions = nr_completions
        self.embedder_model = embedder_model
        self.prompt_template = """Given a question, state which C++ PCL function or class or module can be used to answer it and give a short description. Keep your answer short.

            Question: {{question}}

            Description:
            Minimal Code example:
            """
        self.generator = OllamaChatGenerator(
            model=self.instruct_llm,
            generation_kwargs={"n": self.nr_completions, "num_predict": 256, "temperature": 0.5},
        )
        self.prompt_builder = ChatPromptBuilder(
            template=[ChatMessage.from_user(self.prompt_template)],
        )

        self.adapter = OutputAdapter(
            template="{{answers | build_doc}}",
            output_type=List[Document],
            custom_filters={"build_doc": lambda data: [Document(content=d.content) for d in data]},
            unsafe=True,
        )

        self.embedder = SentenceTransformersDocumentEmbedder(model=embedder_model, progress_bar=True)
        self.embedder.warm_up()

        self.pipeline = Pipeline()
        self.pipeline.add_component(name="prompt_builder", instance=self.prompt_builder)
        self.pipeline.add_component(name="generator", instance=self.generator)
        self.pipeline.add_component(name="adapter", instance=self.adapter)
        self.pipeline.add_component(name="embedder", instance=self.embedder)
        self.pipeline.connect("prompt_builder.prompt", "generator")
        self.pipeline.connect("generator.replies", "adapter.answers")
        self.pipeline.connect("adapter.output", "embedder.documents")

    def to_dict(self) -> Dict[str, Any]:
        data = default_to_dict(
            self,
            instruct_llm=self.instruct_llm,
            instruct_llm_api_key=self.instruct_llm_api_key,
            nr_completions=self.nr_completions,
            embedder_model=self.embedder_model,
        )
        data["pipeline"] = self.pipeline.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HypotheticalDocumentEmbedder":
        hyde_obj = default_from_dict(cls, data)
        hyde_obj.pipeline = Pipeline.from_dict(data["pipeline"])
        return hyde_obj

    @component.output_types(hypothetical_embedding=List[float])
    def run(self, text: str, template: Optional[List[ChatMessage]]):
        if template:
            template.pop()
            template.append(ChatMessage.from_user(self.prompt_template))
        result = self.pipeline.run(data={"prompt_builder": {"template_variables": {"question": text}, "template": template}})
        # return a single query vector embedding representing the average of the hypothetical document embeddings
        stacked_embeddings = array([doc.embedding for doc in result["embedder"]["documents"]])
        avg_embeddings = mean(stacked_embeddings, axis=0)
        hyde_vector = avg_embeddings.reshape((1, len(avg_embeddings)))
        return {"hypothetical_embedding": hyde_vector[0].tolist()}
    

if __name__ == "__main__":
    # Example usage of the HypotheticalDocumentEmbedder component
    hyde = HypotheticalDocumentEmbedder()
    result = hyde.run(text="How to sort a list of numbers in Python?")
    print(result)