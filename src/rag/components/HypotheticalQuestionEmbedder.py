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
        """Initializes the HypotheticalQuestionEmbedder class.

        Args:
            num_questions (int, optional): The number of hypothetical questions to generate. Defaults to 3.
            generator_model (str, optional): The model to use for generating questions. Defaults to "llama3.1".
            embedder_model (str, optional): The model to use for embedding documents. Defaults to "sentence-transformers/all-mpnet-base-v2".

        Attributes:
            generator_model (str): The model used for generating questions.
            embedder_model (str): The model used for embedding documents.
            num_questions (int): The number of hypothetical questions to generate.
            loop_progress (int): Tracks the progress of the loop.
            total_docs (int): The total number of documents processed.
            pipeline (Pipeline): The processing pipeline.
            generator (OllamaGenerator): The generator for creating hypothetical questions.
            prompt_builder (PromptBuilder): The builder for creating prompts.
            adapter (OutputAdapter): The adapter for processing output.
            embedder (SentenceTransformersDocumentEmbedder): The embedder for document embeddings.

        Logs:
            Info: Initialization details and pipeline status."""
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
            template="""
Given the following text as part of the description for a {{document.meta["type"]}} in a software documentation:
{{document.content}}

Formulate exactly {{num_questions}} hypothetical questions, which can be derived from the text. Seperate the questions from each other using a semicolon and newline character.

Your answer should be forumlated exactly like this: Question1;\nQuestion2;\nQuestion3 and your answer should contain nothing other than the questions.
            """
        )
        self.adapter = OutputAdapter(
            template="{{questions | question_splitter}}",
            output_type=List[Document],
            custom_filters={"question_splitter": lambda questions: questions[0].split(";")}
        )
        self.embedder = SentenceTransformersDocumentEmbedder(model=self.embedder_model, meta_fields_to_embed=["type"])
        self.embedder.warm_up()
        logger.info(f"Initialized HypotheticalQuestionEmbedder with generator model: {self.generator_model}, embedder model: {self.embedder_model}")
        self._build_pipeline()
        self._connect_components()
        logger.info("Pipeline built and components connected")

    def _build_pipeline(self) -> None:
        """
        Constructs the processing pipeline by adding necessary components.

        This method adds three components to the pipeline:
        1. A generator component.
        2. A prompt builder component.
        3. An adapter component.

        Each component is added with a specific name and its corresponding instance.
        """
        self.pipeline.add_component(name="generator", instance=self.generator)
        self.pipeline.add_component(name="prompt_builder", instance=self.prompt_builder)
        self.pipeline.add_component(name="adapter", instance=self.adapter)

    def _connect_components(self) -> None:
        """
        Connects the components of the pipeline.

        This method establishes the connections between different components
        in the pipeline. Specifically, it connects the 'prompt_builder' to 
        the 'generator' and the 'generator.replies' to the 'adapter.questions'.
        """
        self.pipeline.connect("prompt_builder", "generator")
        self.pipeline.connect("generator.replies", "adapter.questions")

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the HypotheticalQuestionEmbedder instance to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the HypotheticalQuestionEmbedder instance, 
                            including the generator model, embedder model, number of questions, 
                            loop progress, total documents, and the pipeline.
        """
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
        """
        Create an instance of HypotheticalQuestionEmbedder from a dictionary.

        Args:
            data (Dict[str, Any]): A dictionary containing the data to initialize the object.

        Returns:
            HypotheticalQuestionEmbedder: An instance of HypotheticalQuestionEmbedder initialized with the provided data.
        """
        hyqe_obj = default_from_dict(cls, data)
        hyqe_obj.pipeline = Pipeline.from_dict(data["pipeline"])
        return hyqe_obj

    @component.output_types(question_embeddings=List[Document])
    def run(self, documents: List[Document], print_questions: Optional[bool] = False) -> Dict[str, List[Document]]:
        """
        Generate hypothetical questions for a list of documents and embed them.

        Args:
            documents (List[Document]): A list of Document objects to process.
            print_questions (Optional[bool]): If True, print the generated questions. Default is False.

        Returns:
            Dict[str, List[Document]]: A dictionary with a key "question_embeddings" containing a list of Document objects with embedded hypothetical questions.
        """
        logger.info("Generating hypothetical questions...")
        self.total_docs = len(documents)
        output_list: List[Document] = []
        tqdm.write("Generating hypothetical questions...")
        def process_document(document):
            result = self.pipeline.run(data={"prompt_builder": {"document": document, "num_questions": self.num_questions}})
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
        print("Embedding hypothetical questions...")
        time.sleep(15)
        return_list = []
        chunk_size = max(1, len(output_list) // 10)
        output_chunks = [output_list[i:i + chunk_size] for i in range(0, len(output_list), chunk_size)]
        for chunk in output_chunks:
            return_list.extend(self.embedder.run(chunk)["documents"])
        return {"question_embeddings": return_list}
        
    

if __name__ ==  "__main__":
    hyqe = HypotheticalQuestionEmbedder()
    output = hyqe.run([Document(content="The quick brown fox jumps over the lazy dog.", meta={"type": "sentance", "id": 12345})], print_questions=True)
    for doc in output["question_embeddings"]:
        print(doc)
        

    
    
