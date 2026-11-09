import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.deps import DocumentRepoDep, IngestDep, RetrieverDep, SettingsDep, require_api_key
from app.rag.errors import DuplicateDocumentError, IngestError
from app.schemas import DocumentOut, SearchRequest, SearchResponse


router = APIRouter(tags=["documents"], dependencies=[Depends(require_api_key)])


@router.post("/documents", response_model=DocumentOut, status_code=201)
async def upload_document(file: UploadFile, ingest: IngestDep, settings: SettingsDep):
    # Read at most limit + 1 bytes so a huge upload can't fill memory
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"File larger than {settings.max_upload_mb} MB")

    try:
        return await ingest.ingest(file.filename or "upload.txt", data)
    except DuplicateDocumentError as e:
        raise HTTPException(status_code=409, detail=f"Already uploaded as document {e.document_id}")
    except IngestError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(document_id: uuid.UUID, repo: DocumentRepoDep):
    if not await repo.delete(document_id):
        raise HTTPException(status_code=404, detail="Document not found")


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, retriever: RetrieverDep):
    start = time.perf_counter()
    results = await retriever.search(req.query, req.top_k, req.document_ids)
    retrieval_ms = round((time.perf_counter() - start) * 1000)
    return SearchResponse(query=req.query, results=results, retrieval_ms=retrieval_ms)