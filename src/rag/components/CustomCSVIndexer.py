from typing import Dict, Any, List, Optional
from copy import deepcopy
from haystack import component, Document, default_from_dict, default_to_dict, logging
from haystack.components.preprocessors.document_splitter import DocumentSplitter
from tqdm import tqdm
logger = logging.getLogger(__name__)

@component
class CustomCSVIndexer(DocumentSplitter):
    """
    CustomCSVIndexer is a class that extends the DocumentSplitter to handle custom CSV indexing.
    Attributes:
        splitter (DocumentSplitter): An instance of DocumentSplitter initialized with the provided parameters.
    Methods:
        from_dict(cls, data: Dict[str, Any]) -> "CustomCSVIndexer":
            Class method to create an instance of CustomCSVIndexer from a dictionary.
        to_dict(self) -> Dict[str, Any]:
            Instance method to convert the CustomCSVIndexer instance to a dictionary.
        run(self, documents: List[Document]) -> Dict[str, List[Document]]:
            Processes a list of documents, splitting them based on their type and content, and returns a dictionary with the processed documents.
    """
    def __init__(self, split_by: str = "word", split_length: int = 400, split_overlap: int = 50, split_threshold: int = 50):
        """
        Initializes the CustomCSVIndexer with specified parameters for document splitting.

        Args:
            split_by (str): The criterion by which to split the document. Default is "word".
            split_length (int): The length of each split segment. Default is 400.
            split_overlap (int): The overlap between consecutive split segments. Default is 50.
            split_threshold (int): The threshold for splitting. Default is 50.
        """
        self.splitter = DocumentSplitter(split_by=split_by, split_length=split_length, split_overlap=split_overlap, split_threshold=split_threshold)

    def from_dict(cls, data: Dict[str, Any]) -> "CustomCSVIndexer":
        """
        Create an instance of CustomCSVIndexer from a dictionary.

        Args:
            data (Dict[str, Any]): A dictionary containing the data to initialize the CustomCSVIndexer instance.

        Returns:
            CustomCSVIndexer: An instance of CustomCSVIndexer initialized with the provided data.
        """
        return default_from_dict(cls, data)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the CustomCSVIndexer instance to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the CustomCSVIndexer instance.
        """
        return default_to_dict(self)
    
    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Processes and indexes a list of documents.

        This method iterates over the provided documents and processes them based on their type.
        If the document type is "class" or "module", it splits the document content and creates
        new documents from the split content. If the split content is less than 3 chunks, it retains the
        original document. For other document types, it simply adds the document to the output list. The original
        content is stored in the "full_content" field of the document's metadata.

        Args:
            documents (List[Document]): A list of Document objects to be processed and indexed.

        Returns:
            Dict[str, List[Document]]: A dictionary with a single key "documents" containing the list
            of processed Document objects.
        """
        output_list = []
        logger.info("Indexing documents...")
        for doc in tqdm(documents):
            if doc.meta["type"] == "class" or doc.meta["type"] == "module":
                content_list = self.splitter.run(documents=[doc])["documents"]
                if len(content_list) < 3:
                    new_doc = deepcopy(doc)
                    new_doc.meta["full_content"] = doc.content
                    output_list.append(new_doc)
                    continue
                else:
                    for content in content_list:
                        new_doc = deepcopy(doc)
                        new_doc.content = content.content
                        new_doc.meta["full_content"] = doc.content
                        output_list.append(new_doc)
            else:
                new_doc = deepcopy(doc)
                new_doc.meta["full_content"] = doc.content
                output_list.append(new_doc)
        return {"documents": output_list}
