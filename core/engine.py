import time
from typing import List, Dict, Tuple, Optional
import numpy as np
import jieba
from ..config import Config
from ..storage.unified_db import UnifiedStorage
from .embedding import get_embedding_service
from .topic_classifier import get_classifier


class RAGEngine:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.storage = UnifiedStorage(config=self.config)
        self.embedding = get_embedding_service()
        self._doc_vector_cache = {}
        
    async def ingest(self, group_id: str, chunk_text: str, topic: str = None):
        chunk_id = f"chk_{int(time.time()*1000)}_{hash(chunk_text) % 10000}"
        
        if topic is None:
            classifier = get_classifier()
            topic_type, confidence = classifier.classify(chunk_text)
            topic = topic_type.value
        
        vector = self.embedding.encode_single(chunk_text)
        self.storage.insert(chunk_id, group_id, chunk_text, vector, topic)
        
        if len(self._doc_vector_cache) > 0:
            self._doc_vector_cache.clear()
        return chunk_id, topic

    def _rrf_fusion(self, dense_results: List[Tuple[str, float]], 
                    sparse_results: List[Tuple[str, float]], k: int = 60) -> List[str]:
        rrf_scores = {}
        for rank, (doc_id, _) in enumerate(dense_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        for rank, (doc_id, _) in enumerate(sparse_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_results]

    def _keyword_overlap_score(self, query: str, doc_content: str) -> float:
        query_tokens = set(jieba.cut(query.lower()))
        doc_tokens = set(jieba.cut(doc_content.lower()))
        if not query_tokens or not doc_tokens:
            return 0.0
        intersection = len(query_tokens & doc_tokens)
        union = len(query_tokens | doc_tokens)
        return intersection / union if union > 0 else 0.0

    def _rerank(self, query: str, doc_ids: List[str], top_k: int = 3) -> List[str]:
        if not doc_ids:
            return []
        query_vec = self.embedding.encode_single(query)
        scores = []
        for doc_id in doc_ids:
            doc = self.storage.get_document(doc_id)
            if not doc:
                continue
            doc_vec = None
            if doc_id in self._doc_vector_cache:
                doc_vec = self._doc_vector_cache[doc_id]
            else:
                doc_vec = self.embedding.encode_single(doc['content'])
                if len(self._doc_vector_cache) >= 50:
                    self._doc_vector_cache.clear()
                self._doc_vector_cache[doc_id] = doc_vec
            semantic_score = float(np.dot(query_vec, doc_vec))
            keyword_score = self._keyword_overlap_score(query, doc['content'])
            final_score = 0.7 * semantic_score + 0.3 * keyword_score
            scores.append((doc_id, final_score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in scores[:top_k]]

    async def search(self, group_id: str, query: str) -> List[str]:
        query_vec = self.embedding.encode_single(query)
        dense_results = self.storage.search_faiss(
            query_vec, group_id, top_k=Config.RECALL_TOP_K
        )
        sparse_results = self.storage.search_bm25(
            query, group_id, top_k=Config.RECALL_TOP_K
        )
        if not dense_results and not sparse_results:
            return []
        fused_ids = self._rrf_fusion(dense_results, sparse_results, k=Config.RRF_K)
        rerank_candidates = fused_ids[:8]
        final_ids = self._rerank(query, rerank_candidates, top_k=Config.RERANK_TOP_K)
        results = []
        for doc_id in final_ids:
            doc = self.storage.get_document(doc_id)
            if doc:
                results.append(doc['content'])
        return results
    
    def get_stats(self) -> Dict:
        return self.storage.get_stats()
    
    def create_backup(self) -> Optional[str]:
        return self.storage.create_backup()
    
    def rebuild_index(self):
        self.storage.rebuild_index()
        self._doc_vector_cache.clear()
    
    def forget_old_documents(self, days: int = None) -> int:
        return self.storage.forget_old_documents(days)
    
    def delete_document(self, doc_id: str) -> bool:
        result = self.storage.delete_document(doc_id)
        if result and doc_id in self._doc_vector_cache:
            del self._doc_vector_cache[doc_id]
        return result
