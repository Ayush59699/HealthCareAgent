"""CPU-first local semantic embeddings; never a generative LLM or fallback model."""
import logging
from .config import PatientRAGConfig

logger = logging.getLogger(__name__)


class EmbeddingModel:
    def __init__(self, model_name=None, device=None, batch_size=None, *, allow_download=False):
        config = PatientRAGConfig()
        self.model_name = model_name or config.model_name
        self.device = device or config.device
        self.batch_size = config.batch_size if batch_size is None else batch_size
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError('batch_size must be positive')
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device,
                local_files_only=not allow_download, trust_remote_code=False)
        except Exception as exc:
            raise RuntimeError(
                f'Cannot load embedding model {self.model_name!r} on {self.device}. '
                'Install requirements.txt. To explicitly download the selected model, run '
                'python scripts/build_patient_index.py --limit 1000 --allow-download. '
                'Alternatively set EMBEDDING_MODEL to a complete local model directory. '
                'No substitute model was loaded. Original error: ' + str(exc)) from exc
        dimension_method = getattr(self.model, 'get_embedding_dimension', None)
        if dimension_method is None:
            dimension_method = self.model.get_sentence_embedding_dimension
        self.dimension = dimension_method()
        if not isinstance(self.dimension, int) or self.dimension < 1:
            raise RuntimeError('Embedding model did not report a valid dimension')
        self.truncated_texts = 0
        logger.info('Embedding model=%s device=%s dimension=%d max_tokens=%s',
                    self.model_name, self.device, self.dimension, self.model.max_seq_length)

    @property
    def signature(self):
        # Stored separately from patient payload; prevents mixing embedding spaces.
        return {'model_name': self.model_name, 'dimension': self.dimension,
                'normalize_embeddings': True, 'max_seq_length': self.model.max_seq_length,
                'text_format': 'phase1-to_text-v1', 'prompt': ''}

    def embed_texts(self, texts):
        texts = list(texts)
        if not all(isinstance(t, str) and t.strip() for t in texts):
            raise ValueError('Embedding inputs must be nonempty text strings')
        if not texts:
            return []
        # BGE has a finite context window. Report truncation rather than hiding it.
        for start in range(0, len(texts), self.batch_size):
            lengths = self.model.tokenizer(texts[start:start+self.batch_size],
                truncation=False, padding=False, return_length=True)['length']
            truncated = sum(n > self.model.max_seq_length for n in lengths)
            self.truncated_texts += truncated
            if truncated:
                logger.warning('%d texts exceed model context (%d tokens); embeddings are truncated, payload text is intact',
                               truncated, self.model.max_seq_length)
        vectors = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False, prompt='')
        return vectors.tolist()

    def embed_text(self, text):
        return self.embed_texts([text])[0]
