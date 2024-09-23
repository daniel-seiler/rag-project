from pathlib import Path
from typing import List

from haystack.components.writers import DocumentWriter
from haystack.components.converters import  PyPDFToDocument, MarkdownToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.routers import MetadataRouter

from haystack import Pipeline
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from rag.components.HypotheticalQuestionEmbedder import HypotheticalQuestionEmbedder
from rag.components.CodeCSVToDocument import CodeCSVToDocument
from rag.components.CustomCSVIndexer import CustomCSVIndexer

class IndexerPipeline():
    """
        Initializes the IndexerPipeline with the specified parameters and sets up the necessary components.
        Args:
            mime_types (List[str], optional): List of MIME types to be processed. Defaults to ["text/plain", "application/pdf", "text/markdown", "text/csv"].
            split_by (str, optional): Criterion for splitting documents. Defaults to "sentence".
            split_length (int, optional): Length of each split segment. Defaults to 4.
            split_overlap (int, optional): Overlap between split segments. Defaults to 2.
            split_threshold (int, optional): Threshold for splitting documents. Defaults to 2.
            embedder_model (str, optional): Model to be used for embedding documents. Defaults to 'sentence-transformers/all-mpnet-base-v2'.
            generator_model (str, optional): Model to be used for generating hypothetical questions. Defaults to "llama3.1".
            num_questions (str, optional): Number of questions to generate. Defaults to 3.
        Attributes:
            mime_types (List[str]): List of MIME types to be processed.
            document_store (QdrantDocumentStore): Document store for storing processed documents.
            file_type_router (FileTypeRouter): Router for handling different file types.
            document_cleaner (DocumentCleaner): Cleaner for preprocessing documents.
            document_joiner (DocumentJoiner): Joiner for combining document segments.
            pdf_converter (PyPDFToDocument): Converter for PDF files.
            csv_converter (CodeCSVToDocument): Converter for CSV files.
            markdown_converter (MarkdownToDocument): Converter for Markdown files.
            text_converter (TextFileToDocument): Converter for text files.
            csv_indexer (CustomCSVIndexer): Indexer for CSV files with custom split parameters.
            metadata_router (MetadataRouter): Router for handling metadata based on rules.
            document_splitter (DocumentSplitter): Splitter for dividing documents into segments.
            hyqe_embedder (HypotheticalQuestionEmbedder): Embedder for generating hypothetical questions.
            document_embedder (SentenceTransformersDocumentEmbedder): Embedder for document embeddings.
            document_writer (DocumentWriter): Writer for storing documents in the document store.
            pre_processing_pipeling (Pipeline): Preprocessing pipeline.
            pipeline_status_done (bool): Status flag indicating if the pipeline setup is complete.
        Methods:
            build_pipeline: Constructs the processing pipeline.
            connect_components: Connects the components of the pipeline.
        """
    
    def __init__(self, mime_types:List[str] = ["text/plain", "application/pdf", "text/markdown", "text/csv"], split_by:str="sentence", split_length:int = 4, 
                 split_overlap:int = 2, split_threshold:int = 2, embedder_model:str= 'sentence-transformers/all-mpnet-base-v2', generator_model:str="llama3.1", num_questions:str=3):
        
        self.mime_types = mime_types
        self.document_store = QdrantDocumentStore(url="http://localhost:6333", recreate_index=True)
        self.file_type_router = FileTypeRouter(mime_types=self.mime_types)
        self.document_cleaner = DocumentCleaner()
        self.document_joiner = DocumentJoiner()
        self.pdf_converter = PyPDFToDocument()
        self.csv_converter = CodeCSVToDocument()
        self.markdown_converter = MarkdownToDocument()
        self.text_converter = TextFileToDocument()
        self.csv_indexer = CustomCSVIndexer(split_length=200, split_overlap=25, split_threshold=25)
        self.metadata_router = MetadataRouter(rules={
            "text/csv": {"field": "meta.file_type", "operator": "==", "value": "csv"},
            "others": {"field": "meta.file_type", "operator": "!=", "value": "csv"}
        })
        self.document_splitter = DocumentSplitter(split_by=split_by, split_length=split_length, split_overlap=split_overlap, split_threshold=split_threshold)
        self.hyqe_embedder = HypotheticalQuestionEmbedder(embedder_model=embedder_model, generator_model=generator_model, num_questions=num_questions)
        self.document_embedder = SentenceTransformersDocumentEmbedder(model=embedder_model, meta_fields_to_embed=["type"])
        self.document_writer = DocumentWriter(document_store=self.document_store)
        self.pre_processing_pipeling = Pipeline()
        self.pipeline_status_done = False
        
        self.build_pipeline()
        self.connect_components()
    
    def build_pipeline(self) -> None:
        """
        Builds the pre-processing pipeline by adding various components in a specific order.
        
        The components added to the pipeline include:
        - FileTypeRouter: Routes files based on their type.
        - PDFConverter: Converts PDF files to a suitable format.
        - MarkdownConverter: Converts Markdown files to a suitable format.
        - TextConverter: Converts text files to a suitable format.
        - CSVConverter: Converts CSV files to a suitable format.
        - DocumentJoiner: Joins multiple documents into one.
        - DocumentCleaner: Cleans the document content.
        - MetadataRouter: Routes metadata appropriately.
        - DocumentSplitter: Splits documents into smaller parts.
        - CSVIndexer: Indexes CSV files.
        - Rejoiner: Rejoins documents after processing.
        - DocumentEmbedder: Embeds documents for further processing.
        - DocumentWriter: Writes the processed documents to the desired output.
        
        Note: The HyQEEmbedder component is currently commented out.
        """
        self.pre_processing_pipeling.add_component(instance=self.file_type_router, name="FileTypeRouter")
        self.pre_processing_pipeling.add_component(instance=self.pdf_converter, name="PDFConverter")
        self.pre_processing_pipeling.add_component(instance=self.markdown_converter, name="MarkdownConverter")
        self.pre_processing_pipeling.add_component(instance=self.text_converter, name="TextConverter")
        self.pre_processing_pipeling.add_component(instance=self.csv_converter, name="CSVConverter")
        self.pre_processing_pipeling.add_component(instance=self.document_joiner, name="DocumentJoiner")
        self.pre_processing_pipeling.add_component(instance=self.document_cleaner, name="DocumentCleaner")
        self.pre_processing_pipeling.add_component(instance=self.metadata_router, name="MetadataRouter")
        self.pre_processing_pipeling.add_component(instance=self.document_splitter, name="DocumentSplitter")
        self.pre_processing_pipeling.add_component(instance=self.csv_indexer, name="CSVIndexer")
        self.pre_processing_pipeling.add_component(instance=DocumentJoiner(), name="Rejoiner")
        self.pre_processing_pipeling.add_component(instance=self.document_embedder, name="DocumentEmbedder")
        self.pre_processing_pipeling.add_component(instance=self.document_writer, name="DocumentWriter")

    def connect_components(self) -> None:
        """
        Connects various components of the pre-processing pipeline.

        This method sets up the connections between different components in the 
        pre-processing pipeline to ensure proper data flow. The connections are 
        established based on the type of input files and the required processing 
        steps. The components connected include file type routers, converters, 
        joiners, routers, cleaners, splitters, indexers, embedders, and writers.

        The connections are as follows:
        - Routes PDF files to the PDFConverter.
        - Routes Markdown files to the MarkdownConverter.
        - Routes plain text files to the TextConverter.
        - Routes CSV files to the CSVConverter.
        - Connects all converters to the DocumentJoiner.
        - Routes joined documents to the MetadataRouter.
        - Routes non-CSV documents to the DocumentCleaner.
        - Connects cleaned documents to the DocumentSplitter.
        - Connects split documents to the Rejoiner.
        - Routes CSV documents to the CSVIndexer.
        - Connects indexed CSV documents to the Rejoiner.
        - Connects rejoined documents to the DocumentEmbedder.
        - Connects embedded documents to the DocumentWriter.
        """
        self.pre_processing_pipeling.connect("FileTypeRouter.application/pdf", "PDFConverter.sources")
        self.pre_processing_pipeling.connect("FileTypeRouter.text/markdown", "MarkdownConverter.sources")
        self.pre_processing_pipeling.connect("FileTypeRouter.text/plain", "TextConverter.sources")
        self.pre_processing_pipeling.connect("FileTypeRouter.text/csv", "CSVConverter.sources")
        self.pre_processing_pipeling.connect("PDFConverter", "DocumentJoiner")
        self.pre_processing_pipeling.connect("MarkdownConverter", "DocumentJoiner")
        self.pre_processing_pipeling.connect("TextConverter", "DocumentJoiner")
        self.pre_processing_pipeling.connect("CSVConverter", "DocumentJoiner")
        self.pre_processing_pipeling.connect("DocumentJoiner", "MetadataRouter")
        self.pre_processing_pipeling.connect("MetadataRouter.others", "DocumentCleaner")
        self.pre_processing_pipeling.connect("DocumentCleaner", "DocumentSplitter")
        self.pre_processing_pipeling.connect("DocumentSplitter", "Rejoiner")
        self.pre_processing_pipeling.connect("MetadataRouter.text/csv", "CSVIndexer")
        self.pre_processing_pipeling.connect("CSVIndexer", "Rejoiner")
        self.pre_processing_pipeling.connect("Rejoiner", "DocumentEmbedder")
        self.pre_processing_pipeling.connect("DocumentEmbedder", "DocumentWriter")

    def get_progress(self) -> float:
        """
        Calculate the progress of the hyqe embedding process.

        Returns:
            float: The progress of the embedding process as a float between 0.0 and 1.0.
                   Returns 0.0 if there are no documents to embed.
        """
        if self.hyqe_embedder.total_docs != 0:
            return self.hyqe_embedder.loop_progress/self.hyqe_embedder.total_docs
        else:
            return 0.0
    
    def run(self, folder_path:str) -> None:
        """
        Executes the indexing pipeline on the specified folder path.

        This method performs the following steps:
        1. Draws the pre-processing pipeline diagram and saves it as 'pipeline.png' in the specified folder.
        2. Runs the pre-processing pipeline with the files found in the specified folder.
        3. Sets the pipeline status to done upon completion.

        Args:
            folder_path (str): The path to the folder containing the files to be processed.

        Returns:
            None
        """
        self.pre_processing_pipeling.draw("../docs/preprocessing_pipeline.png")
        self.pre_processing_pipeling.run(
            {
                "FileTypeRouter": {"sources": list(Path(folder_path).glob("**/*"))},
            }
        )
        self.pipeline_status_done = True