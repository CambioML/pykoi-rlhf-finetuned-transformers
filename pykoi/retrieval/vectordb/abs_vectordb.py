import os
import docx2txt

from abc import ABC, abstractmethod
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from pdfminer.high_level import extract_text


class AbsVectorDb(ABC):
    def __init__(self):
        self._texts = []
        self._file_names = []
        self._indexed_file_names = self._get_file_names()

    @property
    def vector_db(self):
        return self._vector_db

    def _extract_from_file(self, file):
        if file.suffix == ".pdf":
            # Logic to extract from PDF (not provided in the original code)
            if file.name not in self._indexed_file_names:
                text = extract_text(file)
                self._texts.append(text)
                self._file_names.append(file.name)
        elif file.suffix == ".docx":
            if file.name not in self._indexed_file_names:
                text = docx2txt.process(file)
                self._texts.append(text)
                self._file_names.append(file.name)
        elif file.suffix == ".txt":
            if file.name not in self._indexed_file_names:
                with open(file, "r") as f:
                    text = f.read()
                self._texts.append(text)
                self._file_names.append(file.name)

    def _extract_from_directory(self, directory, extension):
        for file in directory.glob(f"**/*{extension}"):
            self._extract_from_file(file)

    def _split(self):
        # Here we split the documents, as needed, into smaller chunks.
        # We do this due to the context limits of the LLMs.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=500, length_function=len
        )

        split_texts = []
        metadatas = []

        for file_name, text in zip(self._file_names, self._texts):
            splits = text_splitter.split_text(text)
            split_texts.extend(splits)
            metadatas.extend([{"file_name": file_name}] * len(splits))
        return split_texts, metadatas

    def extract(self):
        doc_path = os.getenv("DOC_PATH")
        path_obj = Path(doc_path)
        if path_obj.is_file():
            # Handle the case where DOC_PATH is a file
            self._extract_from_file(path_obj)
        elif path_obj.is_dir():
            # Handle the case where DOC_PATH is a directory
            for ext in [".pdf", ".docx", ".txt"]:
                self._extract_from_directory(path_obj, ext)
        else:
            # Handle the case where DOC_PATH is neither a file nor a directory
            print("DOC_PATH is neither a file nor a directory!")
        return self._split()

    def index(self):
        split_texts, metadatas = self.extract()
        if self._file_names:
            print(f"Indexing {self._file_names}...")
            self._indexed_file_names = self._indexed_file_names.union(
                set(self._file_names)
            )
            self._index(split_texts, metadatas)
            self._persist()
            self._reset()
        else:
            print("No new files to index.")
        print(f"Already indexed file names: {self._indexed_file_names}")

    def _reset(self):
        self._file_names = []
        self._texts = []

    @abstractmethod
    def _get_file_names(self):
        raise NotImplementedError

    @abstractmethod
    def _index(self, docs, metadatas):
        raise NotImplementedError

    @abstractmethod
    def _persist(self):
        raise NotImplementedError

    @abstractmethod
    def get_embedding(self):
        raise NotImplementedError
