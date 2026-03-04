import numpy as np
from typing import List
import hashlib
from ..config import Config


class EmbeddingService:
    def __init__(self, model_name: str = None, dim: int = None, device: str = None):
        self.dim = dim or Config.EMBEDDING_DIM
        self.model = None
        self._fallback = False
        self._initialized = False
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.device = device or Config.EMBEDDING_DEVICE
        
    
    def _init_model(self):
        if self._initialized:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            print(f"[Embedding] Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.dim = self.model.get_sentence_embedding_dimension()
            print(f"[Embedding] Model dim: {self.dim}")
        except Exception as e:
            print(f"[Embedding] Model load failed, using fallback: {e}")
            self._fallback = True
            self._init_fallback()
        
        self._initialized = True
    
    def _init_fallback(self):
        np.random.seed(42)
        self.projection = np.random.randn(1000, self.dim).astype('float32')
        self.projection /= np.linalg.norm(self.projection, axis=1, keepdims=True)
    
    def _hash_features(self, text: str) -> np.ndarray:
        features = np.zeros(1000, dtype='float32')
        text = text.lower()
        
        for i in range(len(text) - 1):
            bigram = text[i:i+2]
            hash_val = int(hashlib.md5(bigram.encode()).hexdigest(), 16) % 1000
            features[hash_val] += 1
        
        norm = np.linalg.norm(features)
        if norm > 0:
            features /= norm
        
        return features
    
    def encode(self, texts: List[str]) -> np.ndarray:
        # 延迟加载
        if not self._initialized:
            self._init_model()
        
        if self._fallback:
            vectors = []
            for text in texts:
                features = self._hash_features(text)
                vec = features @ self.projection
                vec /= np.linalg.norm(vec)
                vectors.append(vec)
            return np.array(vectors, dtype='float32')
        else:
            vectors = self.model.encode(texts, normalize_embeddings=True)
            return vectors.astype('float32')
    
    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


_embedding_service = None


def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
