from dataclasses import dataclass


@dataclass
class Document:
    abstract: str = ""
    author: str = ""
    medline_ui: str = ""
    mesh_terms: str = ""
    publication_type: str = ""
    seq_id: str = ""
    source: str = ""
    title: str = ""

    @property
    def full_text(self):
        return " ".join([self.title, self.mesh_terms, self.abstract])


@dataclass
class Query:
    num: str = ""
    title: str = ""
    desc: str = ""

    @property
    def full_text(self):
        return self.desc
