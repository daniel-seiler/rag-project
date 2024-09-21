from pathlib import Path
from typing import List

from haystack.components.writers import DocumentWriter
from haystack.components.converters import  PyPDFToDocument, MarkdownToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.routers import MetadataRouter

from haystack import Pipeline
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from components.HypotheticalQuestionEmbedder import HypotheticalQuestionEmbedder
from components.CSVToDocument import CSVToDocument

class IndexerPipeline():
    def __init__(self, mime_types:List[str] = ["text/plain", "application/pdf", "text/markdown", "table/csv"], split_by:str="sentence", split_length:int = 10, 
                 split_overlap:int = 3, split_threshold:int = 3, embedder_model:str= 'sentence-transformers/all-mpnet-base-v2', generator_model:str="llama3.1", num_questions:str=3):
        
        self.mime_types = mime_types
        self.document_store = QdrantDocumentStore(url="http://localhost:6333", recreate_index=True)
        self.file_type_router = FileTypeRouter(mime_types=self.mime_types)
        self.document_cleaner = DocumentCleaner()
        self.document_joiner = DocumentJoiner()
        self.pdf_converter = PyPDFToDocument()
        self.csv_converter = CSVToDocument()
        self.markdown_converter = MarkdownToDocument()
        self.text_converter = TextFileToDocument()
        self.metadata_router = MetadataRouter(rules={
            "table/csv": {"field": "meta.file_type", "operator": "==", "value": "csv"},
            "others": {"field": "meta.file_type", "operator": "!=", "value": "csv"}
        })
        self.document_splitter = DocumentSplitter(split_by=split_by, split_length=split_length, split_overlap=split_overlap, split_threshold=split_threshold)
        self.hyqe_embedder = HypotheticalQuestionEmbedder(embedder_model=embedder_model, generator_model=generator_model, num_questions=num_questions)
        self.document_writer = DocumentWriter(document_store=self.document_store)
        self.pre_processing_pipeling = Pipeline()
        
        self.build_pipeline()
        self.connect_components()
    
    def build_pipeline(self):
        self.pre_processing_pipeling.add_component(instance=self.file_type_router, name="FileTypeRouter")
        self.pre_processing_pipeling.add_component(instance=self.pdf_converter, name="PDFConverter")
        self.pre_processing_pipeling.add_component(instance=self.markdown_converter, name="MarkdownConverter")
        self.pre_processing_pipeling.add_component(instance=self.text_converter, name="TextConverter")
        self.pre_processing_pipeling.add_component(instance=self.csv_converter, name="CSVConverter")
        self.pre_processing_pipeling.add_component(instance=self.document_joiner, name="DocumentJoiner")
        self.pre_processing_pipeling.add_component(instance=self.document_cleaner, name="DocumentCleaner")
        self.pre_processing_pipeling.add_component(instance=self.metadata_router, name="MetadataRouter")
        self.pre_processing_pipeling.add_component(instance=self.document_splitter, name="DocumentSplitter")
        self.pre_processing_pipeling.add_component(instance=DocumentJoiner(), name="Rejoiner")
        self.pre_processing_pipeling.add_component(instance=self.hyqe_embedder, name="HyQEEmbedder")
        self.pre_processing_pipeling.add_component(instance=self.document_writer, name="DocumentWriter")

    def connect_components(self):
        self.pre_processing_pipeling.connect("FileTypeRouter.application/pdf", "PDFConverter.sources")
        self.pre_processing_pipeling.connect("FileTypeRouter.text/markdown", "MarkdownConverter.sources")
        self.pre_processing_pipeling.connect("FileTypeRouter.text/plain", "TextConverter.sources")
        self.pre_processing_pipeling.connect("FileTypeRouter.table/csv", "CSVConverter.sources")
        self.pre_processing_pipeling.connect("PDFConverter", "DocumentJoiner")
        self.pre_processing_pipeling.connect("MarkdownConverter", "DocumentJoiner")
        self.pre_processing_pipeling.connect("TextConverter", "DocumentJoiner")
        self.pre_processing_pipeling.connect("CSVConverter", "DocumentJoiner")
        self.pre_processing_pipeling.connect("DocumentJoiner", "MetadataRouter")
        self.pre_processing_pipeling.connect("MetadataRouter.others", "DocumentCleaner")
        self.pre_processing_pipeling.connect("DocumentCleaner", "DocumentSplitter")
        self.pre_processing_pipeling.connect("DocumentSplitter", "Rejoiner")
        self.pre_processing_pipeling.connect("MetadataRouter.table/csv", "Rejoiner")
        self.pre_processing_pipeling.connect("Rejoiner", "HyQEEmbedder")
        self.pre_processing_pipeling.connect("HyQEEmbedder", "DocumentWriter")
    
    def run(self, folder_path:str):
        self.pre_processing_pipeling.draw(folder_path+"/pipeline.png")
        self.pre_processing_pipeling.run(
            {
                "FileTypeRouter": {"sources": list(Path(folder_path).glob("**/*"))},
            }
        )

    



# def merge_chunks(retrieved_docs, merge_size=300):
#     """
#     Merge smaller chunks progressively into larger chunks.
#     Args:
#     - retrieved_docs: List of retrieved document chunks.
#     - merge_size: int, the maximum number of characters per merged chunk.

#     Returns:
#     - List of merged document chunks.
#     """
#     merged_docs = []
#     temp_chunk = ""
    
#     for doc in retrieved_docs:
#         temp_chunk += doc.content + " "
#         if len(temp_chunk) >= merge_size:
#             merged_docs.append(temp_chunk.strip())
#             temp_chunk = ""
    
#     # Append any leftover content as the final chunk
#     if temp_chunk:
#         merged_docs.append(temp_chunk.strip())
    
#     return merged_docs

# # Merge retrieved small chunks into larger ones
# merged_docs = merge_chunks(retrieved_docs, merge_size=300)

# # Show the merged chunks
# for merged_doc in merged_docs:
#     print(f"Merged chunk: {merged_doc}")