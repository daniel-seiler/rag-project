from haystack import component, Pipeline, Document, default_to_dict, default_from_dict, logging
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.converters import OutputAdapter
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator

from typing import Dict, Any, List, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)


@component
class HypotheticalQuestionEmbedder:
    def __init__(self, num_questions: int = 3, generator_model:str = "llama3.1", embedder_model:str ="sentence-transformers/all-mpnet-base-v2"):
        self.generator_model = generator_model
        self.embedder_model = embedder_model
        self.num_questions = num_questions
        self.loop_progress = 0
        self.total_docs = 0
        self.pipeline = Pipeline()
        self.generator = OllamaGenerator(model=self.generator_model, 
                                         generation_kwargs={"num_predict": -2, "temperature": 0.75,
                                                            "max_tokens": 400},
                                        )
        self.prompt_builder = PromptBuilder(
            template="""Gegeben ist der folgende Text:
            {{text}}
            
            Formuliere genau {{num_questions}} hypothetische Fragen, die aus dem Text abgeleitet werden können. Trenne die Fragen jeweils von einander durch ein semikolon und Zeilenumbruch.
            
            Deine Antwort soll genau so formatiert sein: Frage1;Frage2;Frage3 und nichts außer den Fragen enthalten.
            """
        )
        self.adapter = OutputAdapter(
            template="{{questions | question_splitter}}",
            output_type=List[Document],
            custom_filters={"question_splitter": lambda questions: questions[0].split(";")}
        )
        self.embedder = SentenceTransformersDocumentEmbedder(model=self.embedder_model, progress_bar=True)
        self.embedder.warm_up()
        logger.info(f"Initialized HypotheticalQuestionEmbedder with generator model: {self.generator_model}, embedder model: {self.embedder_model}")
        self._build_pipeline()
        self._connect_components()
        logger.info("Pipeline built and components connected")

    def _build_pipeline(self):
        self.pipeline.add_component(name="generator", instance=self.generator)
        self.pipeline.add_component(name="prompt_builder", instance=self.prompt_builder)
        self.pipeline.add_component(name="adapter", instance=self.adapter)

    def _connect_components(self):
        self.pipeline.connect("prompt_builder", "generator")
        self.pipeline.connect("generator.replies", "adapter.questions")

    def to_dict(self) -> Dict[str, Any]:
        data = default_to_dict(
            self,
            generator_model=self.generator_model,
            embedder_model=self.embedder_model,
            num_questions=self.num_questions,
            loop_progress=self.loop_progress,
            total_docs=self.total_docs
        )
        data["pipeline"] = self.pipeline.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HypotheticalQuestionEmbedder":
        hyqe_obj = default_from_dict(cls, data)
        hyqe_obj.pipeline = Pipeline.from_dict(data["pipeline"])
        return hyqe_obj

    @component.output_types(question_embeddings=List[Document])
    def run(self, documents: List[Document], print_questions: Optional[bool] = False):
        logger.info("Generating hypothetical questions...")
        self.total_docs = len(documents)
        output_list: List[Document] = []
        tqdm.write("Generating hypothetical questions...")
        def process_document(document):
            result = self.pipeline.run(data={"prompt_builder": {"text": document.content, "num_questions": self.num_questions}})
            questions = result["adapter"]["output"]
            if print_questions:
                print(questions)
            processed_docs = []
            for question in questions:
                meta_info = document.meta.copy()
                meta_info["original_text"] = document.content
                processed_docs.append(Document(content=question.strip(), meta=meta_info))
            return processed_docs

        with tqdm(total=len(documents)) as pbar, ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_document, doc): doc for doc in documents}
            for future in as_completed(futures):
                output_list.extend(future.result())
                self.loop_progress += 1
                pbar.update(1)
        logger.info("Embedding hypothetical questions...")
        time.sleep(5)
        output_list = self.embedder.run(output_list)["documents"]
        return {"question_embeddings": output_list}
        
    

if __name__ ==  "__main__":
    hyqe = HypotheticalQuestionEmbedder()
    output = hyqe.run([Document(content="The quick brown fox jumps over the lazy dog.", meta={"type": "sentance", "id": 12345})], print_questions=True)
    for doc in output["question_embeddings"]:
        print(doc)
        

    
    
