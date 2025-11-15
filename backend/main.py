"""
Ingestion Gateway - FastAPI application
Routes uploads to appropriate processors based on data type
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List
import os
from pathlib import Path

from processors.media_processor import MediaProcessor
from processors.json_analyzer import JSONAnalyzer
from storage.storage_engine import StorageDecisionEngine
from storage.directory_manager import DirectoryManager
from metadata.indexer import MetadataIndexer
from monitoring.logger import setup_logger

app = FastAPI(title="Unified Smart Storage System", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
logger = setup_logger()
<<<<<<< HEAD

# Initialize components with error handling
try:
    media_processor = MediaProcessor()
    logger.info("MediaProcessor initialized")
except Exception as e:
    logger.error(f"Failed to initialize MediaProcessor: {e}")
    media_processor = None

try:
    json_analyzer = JSONAnalyzer()
    logger.info("JSONAnalyzer initialized")
except Exception as e:
    logger.error(f"Failed to initialize JSONAnalyzer: {e}")
    json_analyzer = None

try:
    storage_engine = StorageDecisionEngine()
    logger.info("StorageDecisionEngine initialized")
except Exception as e:
    logger.error(f"Failed to initialize StorageDecisionEngine: {e}")
    storage_engine = None

try:
    directory_manager = DirectoryManager()
    logger.info("DirectoryManager initialized")
except Exception as e:
    logger.error(f"Failed to initialize DirectoryManager: {e}")
    directory_manager = None

try:
    metadata_indexer = MetadataIndexer()
    logger.info("MetadataIndexer initialized")
except Exception as e:
    logger.error(f"Failed to initialize MetadataIndexer: {e}")
    metadata_indexer = None
=======
media_processor = MediaProcessor()
json_analyzer = JSONAnalyzer()
storage_engine = StorageDecisionEngine()
directory_manager = DirectoryManager()
metadata_indexer = MetadataIndexer()
>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f

# Ensure upload directory exists
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def detect_mime_type(file: UploadFile) -> str:
    """Detect MIME type from file"""
    content_type = file.content_type
    if not content_type:
        # Fallback: check file extension
        filename = file.filename or ""
        ext = Path(filename).suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".avi": "video/x-msvideo",
            ".json": "application/json",
        }
        content_type = mime_map.get(ext, "application/octet-stream")
    return content_type


def is_media_type(mime_type: str) -> bool:
    """Check if MIME type is media (image/video)"""
    return mime_type.startswith("image/") or mime_type.startswith("video/")


def is_json_type(mime_type: str) -> bool:
    """Check if MIME type is JSON"""
    return mime_type == "application/json" or mime_type.endswith("+json")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "service": "Unified Smart Storage System"}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Unified upload endpoint
    Detects data type and routes to appropriate processor
    """
    try:
        logger.info(f"Received upload: {file.filename}, type: {file.content_type}")
        
        # Detect MIME type
        mime_type = detect_mime_type(file)
        logger.info(f"Detected MIME type: {mime_type}")
        
        # Save uploaded file temporarily
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        result = None
        
        # Route based on data type
        if is_media_type(mime_type):
<<<<<<< HEAD
            if media_processor is None:
                raise HTTPException(
                    status_code=503,
                    detail="Media processor is not available"
                )
=======
>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
            logger.info("Routing to media processor")
            result = await process_media(file_path, mime_type, file.filename)
        
        elif is_json_type(mime_type):
<<<<<<< HEAD
            if json_analyzer is None or storage_engine is None:
                raise HTTPException(
                    status_code=503,
                    detail="JSON processing components are not available"
                )
=======
>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
            logger.info("Routing to JSON analyzer")
            result = await process_json(file_path, file.filename)
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {mime_type}"
            )
        
        # Clean up temp file
        file_path.unlink()
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def process_media(file_path: Path, mime_type: str, filename: str) -> dict:
    """Process media file through media processor"""
    # Generate embeddings and classify
    processing_result = await media_processor.process(file_path, mime_type)
    
    # Get semantic category
    category = processing_result.get("category", "uncategorized")
    
    # Directory manager: create or use existing category directory
<<<<<<< HEAD
    if directory_manager:
        storage_path = await directory_manager.store_media(
            file_path, category, filename
        )
    else:
        # Fallback: use uploads directory
        storage_path = UPLOAD_DIR / filename
        import shutil
        shutil.copy2(file_path, storage_path)
=======
    storage_path = await directory_manager.store_media(
        file_path, category, filename
    )
>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
    
    # Store metadata and index
    metadata = {
        "filename": filename,
        "mime_type": mime_type,
        "category": category,
        "storage_path": str(storage_path),
        "embeddings": processing_result.get("embeddings"),
        "metadata": processing_result.get("metadata", {}),
    }
    
<<<<<<< HEAD
    if metadata_indexer:
        index_id = await metadata_indexer.index_media(metadata)
    else:
        index_id = 0
=======
    index_id = await metadata_indexer.index_media(metadata)
>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
    
    logger.info(f"Media processed: {filename} -> {storage_path}, category: {category}")
    
    return {
        "status": "success",
        "type": "media",
        "filename": filename,
        "category": category,
        "storage_path": str(storage_path),
        "index_id": index_id,
        "metadata": metadata,
    }


async def process_json(file_path: Path, filename: str) -> dict:
    """Process JSON file through JSON analyzer"""
    # Analyze JSON structure
    analysis = await json_analyzer.analyze(file_path)
    
    # Schema builder decides SQL vs NoSQL
    schema_decision = await storage_engine.decide_storage(analysis)
    
    # Store in appropriate database
    storage_result = await storage_engine.store_json(
        file_path, analysis, schema_decision
    )
    
    # Store metadata and index
    metadata = {
        "filename": filename,
        "analysis": analysis,
        "schema_decision": schema_decision,
        "storage_result": storage_result,
    }
    
<<<<<<< HEAD
    if metadata_indexer:
        index_id = await metadata_indexer.index_json(metadata)
    else:
        index_id = 0
=======
    index_id = await metadata_indexer.index_json(metadata)
>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
    
    logger.info(
        f"JSON processed: {filename} -> {schema_decision['storage_type']}, "
        f"schema: {schema_decision.get('schema_name')}"
    )
    
    return {
        "status": "success",
        "type": "json",
        "filename": filename,
        "schema_decision": schema_decision,
        "storage_result": storage_result,
        "index_id": index_id,
    }


@app.post("/api/upload/batch")
async def upload_batch(files: List[UploadFile] = File(...)):
    """Batch upload endpoint"""
    results = []
    for file in files:
        try:
            result = await upload_file(file)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            results.append({
                "status": "error",
                "filename": file.filename,
                "error": str(e),
            })
    return {"results": results}


@app.get("/api/search/media")
async def search_media(
    category: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 20
):
    """Search media by category or semantic query"""
<<<<<<< HEAD
    if metadata_indexer is None:
        raise HTTPException(status_code=503, detail="Metadata indexer is not available")
=======
>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
    results = await metadata_indexer.search_media(
        category=category, query=query, limit=limit
    )
    return {"results": results}


@app.get("/api/search/json")
async def search_json(
    schema: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 20
):
    """Search JSON records"""
<<<<<<< HEAD
    if metadata_indexer is None:
        raise HTTPException(status_code=503, detail="Metadata indexer is not available")
=======
>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
    results = await metadata_indexer.search_json(
        schema=schema, query=query, limit=limit
    )
    return {"results": results}


@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
<<<<<<< HEAD
    if metadata_indexer is None:
        return {"error": "Metadata indexer is not available"}
=======
>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
    stats = await metadata_indexer.get_stats()
    return stats


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

