import csv
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple

from haystack import component, Document, default_from_dict, default_to_dict, logging
from haystack.dataclasses import ByteStream

logger = logging.getLogger(__name__)

@component
class CodeCSVToDocument:
    """
    A class to convert CSV files containing code and other types of data into Document objects.
    Attributes:
        encoding (str): The encoding used to read the CSV files. Default is "utf-8".
        code_storage_dict (Dict[str, str]): A dictionary to store code descriptions temporarily.
        needs_code_dict (Dict[str, Document]): A dictionary to temporarily store Document objects that need code descriptions.
    Methods:
        to_dict() -> Dict[str, Any]:
            Converts the instance to a dictionary.
        from_dict(cls, data: Dict[str, Any]) -> "CodeCSVToDocument":
            Creates an instance of the class from a dictionary.
        run(sources: List[Union[str, Path, ByteStream]], meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None) -> Dict[str, List[Document]]:
            Converts CSV files to Document objects.
    """
    def __init__(self, encoding:str = "utf-8"):
        """
        Initializes the CodeCSVToDocument instance.

        Args:
            encoding (str): The encoding format to be used for reading files. Defaults to "utf-8".

        Attributes:
            encoding (str): Stores the encoding format.
            code_storage_dict (Dict[str, str]): A dictionary to store code snippets with their identifiers.
            needs_code_dict (Dict[str, Document]): A dictionary to store Document objects that need code snippets.
        """
        self.encoding = encoding
        self.code_storage_dict: Dict[str, str] = dict()
        self.needs_code_dict: Dict[str, Document] = dict()

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the object to a dictionary representation.

        Returns:
            Dict[str, Any]: A dictionary containing the object's data.
        """
        return default_to_dict(self, encoding=self.encoding)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CodeCSVToDocument":
        """
        Creates an instance of CodeCSVToDocument from a dictionary.

        Args:
            data (Dict[str, Any]): A dictionary containing the data to create the instance.

        Returns:
            CodeCSVToDocument: An instance of CodeCSVToDocument created from the provided dictionary.
        """
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, sources: List[Union[str, Path, ByteStream]], meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None) -> Dict[str, List[Document]]:
        """
        Converts CSV files to a list of Document objects.

        Args:
            sources (List[Union[str, Path, ByteStream]]): A list of file paths or streams to CSV files.
            meta (Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]): Optional metadata to be included in the documents.

        Returns:
            Dict[str, List[Document]]: A dictionary with a single key "documents" containing a list of Document objects.
        """
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
                            document.content += "\nCode: " + self.code_storage_dict[row.get("name")]
                            documents.append(document)
                        else:
                            self.needs_code_dict[row.get("name")] = Document(content=content, meta=meta_vals)
                if len(self.needs_code_dict) > 0:
                    documents.extend(list(self.needs_code_dict.values()))
        return {"documents": documents}

if __name__ == "__main__":
    converter = CodeCSVToDocument()
    results = converter.run(sources=["scraper/data.csv"])
    documents = results["documents"]
    print(documents[1].content)
    print(documents[1].meta)