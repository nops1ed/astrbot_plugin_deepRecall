import os
import shutil
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class Config:
    BUFFER_MAX_CHARS: int = 400
    BUFFER_IDLE_TIMEOUT: int = 180
    MSG_MIN_LENGTH: int = 3
    
    RRF_K: int = 60
    RECALL_TOP_K: int = 10
    RERANK_TOP_K: int = 3
    
    EMBEDDING_MODEL: str = "BAAI/bge-small-zh"
    EMBEDDING_DIM: int = 512
    EMBEDDING_DEVICE: str = "cpu"
    
    RERANK_MODEL: str = "BAAI/bge-reranker-base"
    ENABLE_RERANKER: bool = False
    
    LLM_PROVIDER: Optional[str] = None
    LLM_MODEL: Optional[str] = None
    LLM_API_KEY: Optional[str] = None
    LLM_API_BASE: Optional[str] = None
    
    MILVUS_URI: str = "http://localhost:19530"
    MILVUS_COLLECTION: str = "group_chat_memory"
    USE_MILVUS: bool = False
    
    STORAGE_DB_PATH: str = "./rag_memory.db"
    
    MIN_VECTOR_COUNT_FOR_IVF: int = 200
    IVF_NLIST: int = 10
    
    ENABLE_AUTO_FORGET: bool = True
    FORGET_DAYS: int = 30
    MAX_DOCUMENTS: int = 10000
    
    ENABLE_BACKUP: bool = True
    BACKUP_DIR: str = "./backups"
    MAX_BACKUPS: int = 5
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        config = cls()
        
        config.BUFFER_MAX_CHARS = int(config_dict.get("BUFFER_MAX_CHARS", config.BUFFER_MAX_CHARS))
        config.BUFFER_IDLE_TIMEOUT = int(config_dict.get("BUFFER_IDLE_TIMEOUT", config.BUFFER_IDLE_TIMEOUT))
        config.MSG_MIN_LENGTH = int(config_dict.get("MSG_MIN_LENGTH", config.MSG_MIN_LENGTH))
        
        config.RRF_K = int(config_dict.get("RRF_K", config.RRF_K))
        config.RECALL_TOP_K = int(config_dict.get("RECALL_TOP_K", config.RECALL_TOP_K))
        config.RERANK_TOP_K = int(config_dict.get("RERANK_TOP_K", config.RERANK_TOP_K))
        
        config.EMBEDDING_MODEL = config_dict.get("EMBEDDING_MODEL", config.EMBEDDING_MODEL)
        config.EMBEDDING_DIM = int(config_dict.get("EMBEDDING_DIM", config.EMBEDDING_DIM))
        config.EMBEDDING_DEVICE = config_dict.get("EMBEDDING_DEVICE", config.EMBEDDING_DEVICE)
        
        config.RERANK_MODEL = config_dict.get("RERANK_MODEL", config.RERANK_MODEL)
        config.ENABLE_RERANKER = bool(config_dict.get("ENABLE_RERANKER", config.ENABLE_RERANKER))
        
        config.LLM_PROVIDER = config_dict.get("LLM_PROVIDER", config.LLM_PROVIDER)
        config.LLM_MODEL = config_dict.get("LLM_MODEL", config.LLM_MODEL)
        config.LLM_API_KEY = config_dict.get("LLM_API_KEY", config.LLM_API_KEY)
        config.LLM_API_BASE = config_dict.get("LLM_API_BASE", config.LLM_API_BASE)
        
        config.MILVUS_URI = config_dict.get("MILVUS_URI", config.MILVUS_URI)
        config.MILVUS_COLLECTION = config_dict.get("MILVUS_COLLECTION", config.MILVUS_COLLECTION)
        config.USE_MILVUS = bool(config_dict.get("USE_MILVUS", config.USE_MILVUS))
        
        config.STORAGE_DB_PATH = config_dict.get("STORAGE_DB_PATH", config.STORAGE_DB_PATH)
        
        config.MIN_VECTOR_COUNT_FOR_IVF = int(config_dict.get("MIN_VECTOR_COUNT_FOR_IVF", config.MIN_VECTOR_COUNT_FOR_IVF))
        config.IVF_NLIST = int(config_dict.get("IVF_NLIST", config.IVF_NLIST))
        
        config.ENABLE_AUTO_FORGET = bool(config_dict.get("ENABLE_AUTO_FORGET", config.ENABLE_AUTO_FORGET))
        config.FORGET_DAYS = int(config_dict.get("FORGET_DAYS", config.FORGET_DAYS))
        config.MAX_DOCUMENTS = int(config_dict.get("MAX_DOCUMENTS", config.MAX_DOCUMENTS))
        
        config.ENABLE_BACKUP = bool(config_dict.get("ENABLE_BACKUP", config.ENABLE_BACKUP))
        config.BACKUP_DIR = config_dict.get("BACKUP_DIR", config.BACKUP_DIR)
        config.MAX_BACKUPS = int(config_dict.get("MAX_BACKUPS", config.MAX_BACKUPS))
        
        return config
    
    @classmethod
    def from_env(cls):
        config = cls()
        
        config.BUFFER_MAX_CHARS = int(os.getenv("BUFFER_MAX_CHARS", config.BUFFER_MAX_CHARS))
        config.BUFFER_IDLE_TIMEOUT = int(os.getenv("BUFFER_IDLE_TIMEOUT", config.BUFFER_IDLE_TIMEOUT))
        config.MSG_MIN_LENGTH = int(os.getenv("MSG_MIN_LENGTH", config.MSG_MIN_LENGTH))
        
        config.RRF_K = int(os.getenv("RRF_K", config.RRF_K))
        config.RECALL_TOP_K = int(os.getenv("RECALL_TOP_K", config.RECALL_TOP_K))
        config.RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", config.RERANK_TOP_K))
        
        config.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", config.EMBEDDING_MODEL)
        config.EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", config.EMBEDDING_DIM))
        config.EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", config.EMBEDDING_DEVICE)
        
        config.RERANK_MODEL = os.getenv("RERANK_MODEL", config.RERANK_MODEL)
        config.ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "false").lower() == "true"
        
        config.LLM_PROVIDER = os.getenv("LLM_PROVIDER", config.LLM_PROVIDER)
        config.LLM_MODEL = os.getenv("LLM_MODEL", config.LLM_MODEL)
        config.LLM_API_KEY = os.getenv("LLM_API_KEY", config.LLM_API_KEY)
        config.LLM_API_BASE = os.getenv("LLM_API_BASE", config.LLM_API_BASE)
        
        config.MILVUS_URI = os.getenv("MILVUS_URI", config.MILVUS_URI)
        config.MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", config.MILVUS_COLLECTION)
        config.USE_MILVUS = os.getenv("USE_MILVUS", "false").lower() == "true"
        
        config.STORAGE_DB_PATH = os.getenv("STORAGE_DB_PATH", config.STORAGE_DB_PATH)
        
        config.MIN_VECTOR_COUNT_FOR_IVF = int(os.getenv("MIN_VECTOR_COUNT_FOR_IVF", config.MIN_VECTOR_COUNT_FOR_IVF))
        config.IVF_NLIST = int(os.getenv("IVF_NLIST", config.IVF_NLIST))
        
        config.ENABLE_AUTO_FORGET = os.getenv("ENABLE_AUTO_FORGET", "true").lower() == "true"
        config.FORGET_DAYS = int(os.getenv("FORGET_DAYS", config.FORGET_DAYS))
        config.MAX_DOCUMENTS = int(os.getenv("MAX_DOCUMENTS", config.MAX_DOCUMENTS))
        
        config.ENABLE_BACKUP = os.getenv("ENABLE_BACKUP", "true").lower() == "true"
        config.BACKUP_DIR = os.getenv("BACKUP_DIR", config.BACKUP_DIR)
        config.MAX_BACKUPS = int(os.getenv("MAX_BACKUPS", config.MAX_BACKUPS))
        
        return config
    
    def create_backup(self, db_path: str) -> Optional[str]:
        if not self.ENABLE_BACKUP:
            return None
        
        os.makedirs(self.BACKUP_DIR, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.BACKUP_DIR, f"rag_memory_backup_{timestamp}.db")
        
        try:
            shutil.copy2(db_path, backup_path)
            self._cleanup_old_backups()
            return backup_path
        except Exception as e:
            print(f"[Config] Backup failed: {e}")
            return None
    
    def _cleanup_old_backups(self):
        try:
            backups = []
            for filename in os.listdir(self.BACKUP_DIR):
                if filename.startswith("rag_memory_backup_") and filename.endswith(".db"):
                    filepath = os.path.join(self.BACKUP_DIR, filename)
                    backups.append((os.path.getmtime(filepath), filepath))
            
            backups.sort(reverse=True)
            
            for _, filepath in backups[self.MAX_BACKUPS:]:
                try:
                    os.remove(filepath)
                except:
                    pass
        except Exception as e:
            print(f"[Config] Failed to cleanup old backups: {e}")
