"""
Metadata + Indexing Layer
Fast lookup for all stored files/records
Enables efficient retrieval
"""
# smart file tracker
# It doesn’t store the actual files
# it stores metadata about them so you can search, filter, and manage them efficiently.
import os
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import json

from storage.database.sql_storage import SQLStorage
from storage.database.nosql_storage import NoSQLStorage

logger = logging.getLogger(__name__)

class MetadataIndexer:
    """Indexes and retrieves metadata for all stored content"""
    
    def __init__(self):
        # Use SQL for metadata index (fast lookups)
        self.sql_storage = SQLStorage()
        self.nosql_storage = NoSQLStorage()
        self._tables_created = False
        self._media_schema = None
        self._json_schema = None
        self._document_schema = None
        self._ensure_metadata_tables()
    
    def _ensure_metadata_tables(self):
        """Ensure metadata index tables exist"""
        # Media metadata table
        media_schema = {
            "table_name": "media_index",
            "columns": [
                {"name": "id", "type": "SERIAL PRIMARY KEY", "nullable": False},
                {"name": "filename", "type": "VARCHAR(255)", "nullable": False},
                {"name": "mime_type", "type": "VARCHAR(100)", "nullable": False},
                {"name": "category", "type": "VARCHAR(100)", "nullable": False},
                {"name": "storage_path", "type": "TEXT", "nullable": False},
                {"name": "embeddings", "type": "JSONB", "nullable": True},
                {"name": "metadata", "type": "JSONB", "nullable": True},
                {"name": "created_at", "type": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP", "nullable": False},
            ],
            "indexes": [
                {"field": "category", "type": "BTREE"},
                {"field": "filename", "type": "BTREE"},
            ],
        }
        
        # JSON metadata table
        json_schema = {
            "table_name": "json_index",
            "columns": [
                {"name": "id", "type": "SERIAL PRIMARY KEY", "nullable": False},
                {"name": "filename", "type": "VARCHAR(255)", "nullable": False},
                {"name": "schema_name", "type": "VARCHAR(255)", "nullable": False},
                {"name": "storage_type", "type": "VARCHAR(50)", "nullable": False},
                {"name": "storage_location", "type": "TEXT", "nullable": False},
                {"name": "analysis", "type": "JSONB", "nullable": True},
                {"name": "created_at", "type": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP", "nullable": False},
            ],
            "indexes": [
                {"field": "schema_name", "type": "BTREE"},
                {"field": "storage_type", "type": "BTREE"},
            ],
        }
        
        # Document metadata table
        document_schema = {
            "table_name": "document_index",
            "columns": [
                {"name": "id", "type": "SERIAL PRIMARY KEY", "nullable": False},
                {"name": "filename", "type": "VARCHAR(255)", "nullable": False},
                {"name": "mime_type", "type": "VARCHAR(100)", "nullable": False},
                {"name": "category", "type": "VARCHAR(100)", "nullable": False},
                {"name": "storage_path", "type": "TEXT", "nullable": False},
                {"name": "text", "type": "TEXT", "nullable": True},
                {"name": "embeddings", "type": "JSONB", "nullable": True},
                {"name": "metadata", "type": "JSONB", "nullable": True},
                {"name": "created_at", "type": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP", "nullable": False},
            ],
            "indexes": [
                {"field": "category", "type": "BTREE"},
                {"field": "filename", "type": "BTREE"},
                {"field": "mime_type", "type": "BTREE"},
            ],
        }
        
        # Create tables (async, but called during init)
        # Use try-except to handle cases where event loop is already running
        import asyncio
        try:
            # Try to get existing event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, schedule for later
                    logger.warning("Event loop is running, tables will be created on first use")
                    self._tables_created = False
                    self._media_schema = media_schema
                    self._json_schema = json_schema
                    self._document_schema = document_schema
                    return
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Create tables
            try:
                loop.run_until_complete(self.sql_storage.create_table(media_schema))
                loop.run_until_complete(self.sql_storage.create_table(json_schema))
                loop.run_until_complete(self.sql_storage.create_table(document_schema))
                self._tables_created = True
            except Exception as e:
                logger.warning(f"Could not create tables during init: {e}. Will retry on first use.")
                self._tables_created = False
                self._media_schema = media_schema
                self._json_schema = json_schema
                self._document_schema = document_schema
        except Exception as e:
            logger.error(f"Error in table initialization: {e}")
            self._tables_created = False
            self._media_schema = media_schema
            self._json_schema = json_schema
            self._document_schema = document_schema
    
    async def _ensure_tables_async(self):
        """Ensure tables are created (async version)"""
        if not self._tables_created and self._media_schema and self._json_schema and self._document_schema:
            try:
                await self.sql_storage.create_table(self._media_schema)
                await self.sql_storage.create_table(self._json_schema)
                await self.sql_storage.create_table(self._document_schema)
                self._tables_created = True
            except Exception as e:
                logger.warning(f"Could not create tables: {e}")
    
    #Add a file
    async def index_media(self, metadata: Dict) -> int:
        """Index media file metadata"""
        await self._ensure_tables_async()
        
        record = {
            "filename": metadata.get("filename"),
            "mime_type": metadata.get("mime_type"),
            "category": metadata.get("category", "uncategorized"),
            "storage_path": str(metadata.get("storage_path", "")),
            "embeddings": json.dumps(metadata.get("embeddings")) if metadata.get("embeddings") else None,
            "metadata": json.dumps(metadata.get("metadata", {})),
        }
        
        result = await self.sql_storage.insert("media_index", record)
        index_id = result.get("id")
        
        logger.info(f"Indexed media: {metadata.get('filename')} -> index_id: {index_id}")
        return index_id or 0
    
    #Add a jsonfile
    async def index_json(self, metadata: Dict) -> int:
        """Index JSON file metadata"""
        await self._ensure_tables_async()
        
        schema_decision = metadata.get("schema_decision", {})
        storage_result = metadata.get("storage_result", {})
        
        schema = schema_decision.get("schema", {})
        record = {
            "filename": metadata.get("filename"),
            "schema_name": schema.get(
                "schema_name",
                schema.get("collection_name", "unknown")
            ),
            "storage_type": schema_decision.get("storage_type", "unknown"),
            "storage_location": (
                storage_result.get("table_name") or
                storage_result.get("collection_name") or
                "unknown"
            ),
            "analysis": json.dumps(metadata.get("analysis", {})),
        }
        
        result = await self.sql_storage.insert("json_index", record)
        index_id = result.get("id")
        
        logger.info(f"Indexed JSON: {metadata.get('filename')} -> index_id: {index_id}")
        return index_id or 0
    
    # simple filtering
    async def search_media(
        self, category: Optional[str] = None, query: Optional[str] = None, limit: int = 20
    ) -> List[Dict]:
        """Search media by category or query"""
        filters = {}
        if category:
            filters["category"] = category
        
        results = await self.sql_storage.query("media_index", filters, limit)
        
        # TODO: Implement semantic search using embeddings if query is provided
        # For now, return category-filtered results
        
        return results
    
    async def search_json(
        self, schema: Optional[str] = None, query: Optional[str] = None, limit: int = 20
    ) -> List[Dict]:
        """Search JSON records"""
        filters = {}
        if schema:
            filters["schema_name"] = schema
        
        results = await self.sql_storage.query("json_index", filters, limit)
        
        # TODO: Implement query-based search in actual storage
        
        return results
        # simple filtering till here!
    
    async def index_document(self, metadata: Dict) -> int:
        """Index document file metadata"""
        await self._ensure_tables_async()
        
        record = {
            "filename": metadata.get("filename"),
            "mime_type": metadata.get("mime_type"),
            "category": metadata.get("category", "uncategorized"),
            "storage_path": str(metadata.get("storage_path", "")),
            "text": metadata.get("text", "")[:10000],  # Limit text to 10k chars for storage
            "embeddings": json.dumps(metadata.get("embeddings")) if metadata.get("embeddings") else None,
            "metadata": json.dumps(metadata.get("metadata", {})),
        }
        
        result = await self.sql_storage.insert("document_index", record)
        index_id = result.get("id")
        
        logger.info(f"Indexed document: {metadata.get('filename')} -> index_id: {index_id}")
        return index_id or 0
    
    async def search_documents(
        self, category: Optional[str] = None, mime_type: Optional[str] = None,
        query: Optional[str] = None, limit: int = 20
    ) -> List[Dict]:
        """Search documents by category, mime type, or text query"""
        filters = {}
        if category:
            filters["category"] = category
        if mime_type:
            filters["mime_type"] = mime_type
        
        results = await self.sql_storage.query("document_index", filters, limit)
        
        # TODO: Implement full-text search if query is provided
        # For now, filter results by query in text field (basic)
        if query:
            query_lower = query.lower()
            results = [
                r for r in results
                if query_lower in (r.get("text", "") or "").lower()
                or query_lower in (r.get("filename", "") or "").lower()
            ]
        
        return results
    
    async def get_stats(self) -> Dict:
        """Get system statistics"""
        # Count media files
        media_count = len(await self.sql_storage.query("media_index", limit=10000))
        
        # Count JSON files
        json_count = len(await self.sql_storage.query("json_index", limit=10000))
        
        # Count document files
        document_count = len(await self.sql_storage.query("document_index", limit=10000))
        
        # Get categories
        media_records = await self.sql_storage.query("media_index", limit=10000)
        categories = set(r.get("category") for r in media_records if r.get("category"))
        
        document_records = await self.sql_storage.query("document_index", limit=10000)
        doc_categories = set(r.get("category") for r in document_records if r.get("category"))
        categories.update(doc_categories)
        
        # Get schemas
        json_records = await self.sql_storage.query("json_index", limit=10000)
        schemas = set(r.get("schema_name") for r in json_records if r.get("schema_name"))
        
        return {
            "media_files": media_count,
            "json_files": json_count,
            "document_files": document_count,
            "categories": len(categories),
            "schemas": len(schemas),
            "category_list": sorted(list(categories)),
            "schema_list": sorted(list(schemas)),
        }

