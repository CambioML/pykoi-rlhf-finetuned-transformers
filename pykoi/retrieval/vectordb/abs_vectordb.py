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

    def _extract_pdf(self):
        files = list(Path(os.getenv("DOC_PATH")).glob("**/*.pdf"))
        for file in files:
            if file.name not in self._indexed_file_names:
                text = extract_text(file)
                self._texts.append(text)
                self._file_names.append(file.name)

    def _extract_docx(self):
        files = list(Path(os.getenv("DOC_PATH")).glob("**/*.docx"))
        for file in files:
            if file.name not in self._indexed_file_names:
                text = docx2txt.process(file)
                self._texts.append(text)
                self._file_names.append(file.name)

    def _extract_txt(self):
        files = list(Path(os.getenv("DOC_PATH")).glob("**/*.txt"))
        for file in files:
            if file.name not in self._indexed_file_names:
                with open(file, "r") as f:
                    text = f.read()
                self._texts.append(text)
                self._file_names.append(file.name)

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
        self._extract_pdf()
        self._extract_docx()
        self._extract_txt()
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
