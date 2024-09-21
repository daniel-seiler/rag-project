import csv
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple

from haystack import component, Document, default_from_dict, default_to_dict, logging
from haystack.dataclasses import ByteStream
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata

logger = logging.getLogger(__name__)

@component
class CSVToDocument:
    def __init__(self, encoding:str = "utf-8"):
        self.encoding = encoding
        self.code_storage_dict: Dict[str, str] = dict()
        self.needs_code_dict: Dict[str, Document] = dict()

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(self, encoding=self.encoding)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CSVToDocument":
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, sources: List[Union[str, Path, ByteStream]], meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None) -> Dict[str, Any]:
        logger.info("Converting CSV files to documents...")
        documents = []
        for source in sources:
            with open(source, "r", encoding=self.encoding) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("type") == "code":
                        if row.get("name") in self.needs_code_dict.keys():
                            document = self.needs_code_dict[row.get("name")]
                            document.meta["code"] = row.get("description")
                            documents.append(document)
                            self.needs_code_dict.pop(row.get("name"))
                        else:
                            self.code_storage_dict[row.get("name")] = row.get("description")
                    else:
                        content = "Name: " + row.get("name") + "\n" + "Type: " + row.get("type") + "\n" + "Description: " + row.get("description")
                        meta_keys = list(row.keys())
                        meta_keys.remove("name")
                        meta_keys.remove("description")
                        meta_vals = {key: row.get(key) for key in meta_keys}
                        meta_vals["file_type"] = "csv"
                        if row.get("name") in self.code_storage_dict.keys():
                            document = Document(content=content, meta=meta_vals)
                            document.meta["code"] = self.code_storage_dict[row.get("name")]
                            documents.append(document)
                        else:
                            self.needs_code_dict[row.get("name")] = Document(content=content, meta=meta_vals)

        return {"documents": documents}

if __name__ == "__main__":
    converter = CSVToDocument()
    results = converter.run(sources=["scraper/data.csv"])
    documents = results["documents"]
    print(documents[1].content)
    print(documents[1].meta)