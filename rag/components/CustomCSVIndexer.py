from typing import Dict, Any, List, Optional
from copy import deepcopy
from haystack import component, Document, default_from_dict, default_to_dict, logging
from haystack.components.preprocessors.document_splitter import DocumentSplitter
from tqdm import tqdm
logger = logging.getLogger(__name__)

@component
class CustomCSVIndexer(DocumentSplitter):
    def __init__(self, split_by: str = "word", split_length: int = 400, split_overlap: int = 50, split_threshold: int = 50):
        self.splitter = DocumentSplitter(split_by=split_by, split_length=split_length, split_overlap=split_overlap, split_threshold=split_threshold)

    def from_dict(cls, data: Dict[str, Any]) -> "CustomCSVIndexer":
        return default_from_dict(cls, data)
    
    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(self)
    
    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, Any]:
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
