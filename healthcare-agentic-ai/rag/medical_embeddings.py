"""Reuse the Phase 2 BGE implementation, not its patient representation/config."""
from .embeddings import EmbeddingModel


class MedicalEmbeddingModel(EmbeddingModel):
    def __init__(self, *, allow_download=False, device='cpu', batch_size=16):
        super().__init__('BAAI/bge-small-en-v1.5', device, batch_size, allow_download=allow_download)
        if self.dimension != 384:
            raise ValueError('Expected BGE-small embedding dimension 384')

    @property
    def signature(self):
        return {**super().signature, 'text_format': 'medical-source-text-v1',
                'pipeline': 'medical-knowledge-v1'}
