import sqlite3
import time
import shutil
import numpy as np
from typing import List, Dict, Tuple, Optional
from contextlib import contextmanager
from datetime import datetime, timedelta
import faiss
from rank_bm25 import BM25Okapi
import jieba
from ..config import Config


class UnifiedStorage:
    def __init__(self, db_path: str = None, dim: int = None, config: Config = None):
        self.config = config or Config()
        self.db_path = db_path or self.config.STORAGE_DB_PATH
        self.dim = dim or self.config.EMBEDDING_DIM
        
        self.MIN_VECTOR_COUNT_FOR_IVF = self.config.MIN_VECTOR_COUNT_FOR_IVF
        self.IVF_NLIST = self.config.IVF_NLIST
        
        self.index = None
        self.doc_ids = []
        self._use_flat_index = True
        
        self.bm25 = None
        self.bm25_corpus = []
        self.bm25_doc_ids = []
        self._bm25_dirty = False
        
        self._doc_cache = {}
        
        self._init_db()
        self._load_existing_data()
    
    @contextmanager
    def _get_db_connection(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        try:
            yield conn
        finally:
            conn.close()
    
    def _init_db(self):
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    group_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    topic TEXT DEFAULT 'general',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    vector BLOB
                )
            ''')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_group ON documents(group_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_topic ON documents(topic)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_created ON documents(created_at)')
            
            conn.commit()
    
    def _load_existing_data(self):
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id, content, vector FROM documents')
            rows = cursor.fetchall()
        
        if not rows:
            self._init_empty_faiss_index()
            return
        
        self.bm25_corpus = []
        self.bm25_doc_ids = []
        
        vectors = []
        self.doc_ids = []
        
        for doc_id, content, vector_blob in rows:
            tokens = list(jieba.cut(content))
            self.bm25_corpus.append(tokens)
            self.bm25_doc_ids.append(doc_id)
            
            if vector_blob:
                vec = np.frombuffer(vector_blob, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                vectors.append(vec)
                self.doc_ids.append(doc_id)
        
        if self.bm25_corpus:
            self.bm25 = BM25Okapi(self.bm25_corpus)
        
        self._init_faiss_index(vectors)
    
    def _init_empty_faiss_index(self):
        self.index = faiss.IndexFlatIP(self.dim)
        self._use_flat_index = True
    
    def _init_faiss_index(self, vectors: List[np.ndarray]):
        if not vectors:
            self._init_empty_faiss_index()
            return
        
        vectors_np = np.array(vectors).astype('float32')
        
        if len(vectors) >= self.MIN_VECTOR_COUNT_FOR_IVF:
            quantizer = faiss.IndexFlatIP(self.dim)
            nlist = min(self.IVF_NLIST, len(vectors) // 20)
            nlist = max(nlist, 2)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist)
            self.index.train(vectors_np)
            self.index.add(vectors_np)
            self._use_flat_index = False
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(vectors_np)
            self._use_flat_index = True
    
    def _rebuild_bm25_if_needed(self):
        if self._bm25_dirty and self.bm25_corpus:
            self.bm25 = BM25Okapi(self.bm25_corpus)
            self._bm25_dirty = False
    
    def insert(self, doc_id: str, group_id: str, content: str, 
               vector: np.ndarray, topic: str = "general"):
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        vector_blob = vector.astype('float32').tobytes()
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO documents (id, group_id, content, topic, vector)
                VALUES (?, ?, ?, ?, ?)
            ''', (doc_id, group_id, content, topic, vector_blob))
            conn.commit()
        
        tokens = list(jieba.cut(content))
        self.bm25_corpus.append(tokens)
        self.bm25_doc_ids.append(doc_id)
        self._bm25_dirty = True
        
        vector = vector.astype('float32').reshape(1, -1)
        
        if self.index is None:
            self._init_empty_faiss_index()
        
        current_count = len(self.doc_ids)
        if self._use_flat_index and current_count + 1 >= self.MIN_VECTOR_COUNT_FOR_IVF:
            all_vectors = self._load_all_vectors()
            all_vectors.append(vector[0])
            self._init_faiss_index(all_vectors)
            self.doc_ids.append(doc_id)
        else:
            self.index.add(vector)
            self.doc_ids.append(doc_id)
        
        if self.config.ENABLE_AUTO_FORGET:
            self._maybe_trigger_forget()
    
    def _maybe_trigger_forget(self):
        stats = self.get_stats()
        
        if stats['total_documents'] >= self.config.MAX_DOCUMENTS:
            print(f"[Storage] Document limit reached, triggering auto-forget")
            self.forget_old_documents(self.config.FORGET_DAYS)
    
    def forget_old_documents(self, days: int = None) -> int:
        days = days or self.config.FORGET_DAYS
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.strftime('%Y-%m-%d %H:%M:%S')
        
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT id FROM documents WHERE created_at < ?', (cutoff_str,))
            old_ids = [row[0] for row in cursor.fetchall()]
            
            if not old_ids:
                return 0
            
            self.config.create_backup(self.db_path)
            
            cursor.execute('DELETE FROM documents WHERE created_at < ?', (cutoff_str,))
            conn.commit()
            
            deleted_count = cursor.rowcount
            print(f"[Storage] Deleted {deleted_count} old documents")
            
            self.rebuild_index()
            
            return deleted_count
    
    def delete_document(self, doc_id: str) -> bool:
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM documents WHERE id = ?', (doc_id,))
            conn.commit()
            
            if cursor.rowcount > 0:
                self.rebuild_index()
                return True
            return False
    
    def _load_all_vectors(self) -> List[np.ndarray]:
        vectors = []
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT vector FROM documents WHERE vector IS NOT NULL')
            for row in cursor.fetchall():
                if row[0]:
                    vec = np.frombuffer(row[0], dtype=np.float32)
                    vectors.append(vec)
        return vectors
    
    def create_backup(self) -> Optional[str]:
        return self.config.create_backup(self.db_path)
    
    def rebuild_index(self):
        print(f"[Storage] Rebuilding index")
        self._load_existing_data()
        print(f"[Storage] Index rebuilt")
    
    def search_bm25(self, query: str, group_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        self._rebuild_bm25_if_needed()
        
        if not self.bm25:
            return []
        
        query_tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(query_tokens)
        
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM documents WHERE group_id = ?', (group_id,))
            valid_ids = {row[0] for row in cursor.fetchall()}
        
        results = []
        for idx, score in enumerate(scores):
            doc_id = self.bm25_doc_ids[idx]
            if doc_id in valid_ids:
                results.append((doc_id, float(score)))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def search_faiss(self, query_vector: np.ndarray, group_id: str, 
                     top_k: int = 10) -> List[Tuple[str, float]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
        
        query_vector = query_vector.astype('float32').reshape(1, -1)
        
        search_k = min(top_k * 3, self.index.ntotal)
        distances, indices = self.index.search(query_vector, search_k)
        
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM documents WHERE group_id = ?', (group_id,))
            valid_ids = {row[0] for row in cursor.fetchall()}
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.doc_ids):
                continue
            doc_id = self.doc_ids[idx]
            if doc_id in valid_ids:
                results.append((doc_id, float(dist)))
        
        return results[:top_k]
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        if doc_id in self._doc_cache:
            return self._doc_cache[doc_id]
        
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, group_id, content, topic, created_at 
                FROM documents WHERE id = ?
            ''', (doc_id,))
            row = cursor.fetchone()
        
        if row:
            doc = {
                'id': row[0],
                'group_id': row[1],
                'content': row[2],
                'topic': row[3],
                'created_at': row[4]
            }
            if len(self._doc_cache) > 100:
                self._doc_cache.clear()
            self._doc_cache[doc_id] = doc
            return doc
        return None
    
    def get_stats(self) -> Dict:
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM documents')
            total_docs = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(DISTINCT group_id) FROM documents')
            total_groups = cursor.fetchone()[0]
        
        return {
            'total_documents': total_docs,
            'total_groups': total_groups,
            'faiss_index_size': self.index.ntotal if self.index else 0,
            'bm25_corpus_size': len(self.bm25_corpus),
            'index_type': 'Flat' if self._use_flat_index else 'IVF'
        }
