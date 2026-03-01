import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import sys
import os

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import Config
from storage.unified_db import UnifiedStorage
from core.embedding import get_embedding_service


app = FastAPI(title="deepRecall WebUI", version="1.0.0")

config = Config.from_env()
storage = UnifiedStorage(config=config)
embedding = get_embedding_service()


class SearchRequest(BaseModel):
    query: str
    group_id: Optional[str] = None
    top_k: int = 3


class DeleteRequest(BaseModel):
    doc_id: str


class ForgetRequest(BaseModel):
    days: Optional[int] = None


@app.get("/")
async def get_index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/stats")
async def get_stats():
    stats = storage.get_stats()
    return stats


@app.post("/api/search")
async def search(request: SearchRequest):
    query_vec = embedding.encode_single(request.query)
    
    dense_results = storage.search_faiss(query_vec, request.group_id or "", top_k=10)
    sparse_results = storage.search_bm25(request.query, request.group_id or "", top_k=10)
    
    if not dense_results and not sparse_results:
        return {"results": []}
    
    def _rrf_fusion(dense, sparse, k=60):
        rrf_scores = {}
        for rank, (doc_id, _) in enumerate(dense):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        for rank, (doc_id, _) in enumerate(sparse):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_results]
    
    fused_ids = _rrf_fusion(dense_results, sparse_results)
    
    results = []
    for doc_id in fused_ids[:request.top_k]:
        doc = storage.get_document(doc_id)
        if doc:
            results.append(doc)
    
    return {"results": results}


@app.post("/api/forget")
async def forget(request: ForgetRequest):
    deleted = storage.forget_old_documents(request.days)
    return {"deleted_count": deleted}


@app.post("/api/delete")
async def delete(request: DeleteRequest):
    success = storage.delete_document(request.doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"success": True}


@app.post("/api/backup")
async def backup():
    backup_path = config.create_backup(config.STORAGE_DB_PATH)
    if not backup_path:
        raise HTTPException(status_code=500, detail="Backup failed")
    return {"backup_path": backup_path}


@app.post("/api/rebuild")
async def rebuild():
    storage.rebuild_index()
    stats = storage.get_stats()
    return {"success": True, "stats": stats}


if __name__ == "__main__":
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)
    
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    print(f"deepRecall WebUI starting on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
