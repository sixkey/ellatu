import json
import os
from typing import Optional, Dict, List
from .ellatu_db import Document

class DocumentEditor:

    def __init__(self, json: Optional[Document] = None):
        self._doc = json if json is not None else {}

    def inject(self, doc: Document) -> 'DocumentEditor':
        for key, value in doc.items():
            self._doc[key] = value
        return self

    def select(self, keys: List[str]) -> 'DocumentEditor':
        new_doc = {}
        for key in keys:
            if key in self._doc:
                new_doc[key] = self._doc[key]
        self._doc = new_doc
        return self

    def relabel(self, relabling: Dict[str, str]) -> 'DocumentEditor':
        for oldkey, newkey in relabling.items():
            self._doc[newkey] = self._doc[oldkey]
            del self._doc[oldkey]
        return self

    def inject_json_file(self, filename: str,
                         optional: bool = False) -> 'DocumentEditor':
        if not optional or os.path.isfile(filename):
            with open(filename, 'r') as f:
                json_obj = json.load(f)
                for key, value in json_obj.items():
                    self._doc[key] = value
        return self

    def save(self, filename: str) -> 'DocumentEditor':
        with open(filename, 'w') as f:
            json.dump(self._doc, fp=f)
        return self

    def print(self, *args, **kwargs) -> 'DocumentEditor':
        print(json.dumps(self._doc, *args, **kwargs))
        return self

    def mat(self) -> Document:
        return self._doc
