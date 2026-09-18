"""Semantic similarity is a retrieval score, never diagnostic probability."""


class PatientCaseRetriever:
    def __init__(self, embedding_model, vector_store):
        if (embedding_model.dimension != vector_store.dimension or
                embedding_model.signature != vector_store.signature):
            raise ValueError('Embedding configuration mismatch between retriever and vector store')
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(self, patient_text: str, top_k=5):
        if not isinstance(patient_text, str) or not patient_text.strip():
            raise ValueError('patient_text must be nonempty label-free text')
        if type(top_k) is not int or top_k < 0:
            raise ValueError('top_k must be a nonnegative integer')
        if top_k == 0 or self.vector_store.count() == 0:
            return []
        return self.vector_store.search(self.embedding_model.embed_text(patient_text), top_k)
