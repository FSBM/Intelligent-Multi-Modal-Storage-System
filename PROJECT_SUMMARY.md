# Project Summary

## Overview

This is a **Unified Smart Storage System** that intelligently ingests, analyzes, categorizes, stores, and retrieves any type of data: media (images/videos) and structured JSON.

## Key Features

### 1. **Unified Frontend Interface**
- Single web UI for uploading both media and JSON files
- Drag-and-drop file upload
- Real-time upload status and results
- Search interface for retrieving stored content
- Statistics dashboard

### 2. **Intelligent Data Processing**

#### Media Processing
- Automatic MIME type detection
- CLIP/ViT embeddings generation for semantic understanding
- Automatic categorization (nature, animals, people, etc.)
- Metadata extraction (dimensions, format, etc.)

#### JSON Processing
- Structure analysis and pattern detection
- Batch JSON handling with structure merging
- Field consistency analysis
- Value pattern detection (email, URL, date, etc.)

### 3. **Smart Storage Decision**

#### Schema Builder
- Automatically decides SQL vs NoSQL based on:
  - Data consistency (70%+ threshold for SQL)
  - Nesting depth (≤3 for SQL)
  - Field count (≤50 for SQL)
- Auto-generates database schemas
- Creates appropriate indexes

#### Storage Engine
- **SQL (PostgreSQL)**: For consistent, structured data
- **NoSQL (MongoDB)**: For flexible, nested data
- **File Storage**: For media files organized by category

### 4. **Directory Management**
- Automatic category-based directory creation
- Media files stored in `storage/media/{category}/`
- Unique filename generation to prevent conflicts

### 5. **Metadata & Indexing**
- Fast lookup system for all stored content
- Separate indexes for media and JSON
- Category and schema-based search
- Statistics tracking

### 6. **Monitoring & Logging**
- Structured JSON logging
- Tracks all operations (ingestion, storage decisions, errors)
- Health check endpoints

## Architecture

```
Frontend (Next.js)
    ↓
Ingestion Gateway (FastAPI)
    ↓
    ├─→ Media Processor (CLIP/ViT)
    │       ↓
    │   Directory Manager
    │       ↓
    │   File Storage
    │
    └─→ JSON Analyzer
            ↓
        Schema Builder
            ↓
        Storage Decision Engine
            ↓
        SQL (Postgres) or NoSQL (MongoDB)
    ↓
Metadata Indexer
```

## File Structure

```
Multi_Model_Storage/
├── backend/
│   ├── main.py                 # FastAPI gateway
│   ├── processors/
│   │   ├── media_processor.py  # CLIP embeddings & classification
│   │   └── json_analyzer.py    # Structure analysis
│   ├── storage/
│   │   ├── schema_builder.py   # SQL vs NoSQL decision
│   │   ├── storage_engine.py   # Storage orchestration
│   │   ├── directory_manager.py # File organization
│   │   └── database/
│   │       ├── sql_storage.py  # PostgreSQL operations
│   │       └── nosql_storage.py # MongoDB operations
│   ├── metadata/
│   │   └── indexer.py          # Metadata indexing
│   └── monitoring/
│       └── logger.py           # Structured logging
├── frontend/
│   ├── app/                    # Next.js app directory
│   └── components/
│       ├── FileUpload.tsx
│       ├── SearchInterface.tsx
│       └── StatsDashboard.tsx
└── examples/                   # Sample test files
```

## Workflow Examples

### Media Upload Flow
1. User uploads image via web UI
2. Gateway detects MIME type (image/jpeg)
3. Media processor:
   - Generates CLIP embeddings
   - Classifies into category (e.g., "nature")
4. Directory manager:
   - Creates/uses `storage/media/nature/`
   - Stores file with unique name
5. Metadata indexer:
   - Stores metadata in `media_index` table
   - Includes embeddings, category, path
6. Returns confirmation with storage path

### JSON Upload Flow
1. User uploads JSON file
2. Gateway detects MIME type (application/json)
3. JSON analyzer:
   - Extracts structure
   - Detects patterns
   - Calculates consistency
4. Schema builder:
   - Decides SQL (consistent) or NoSQL (flexible)
   - Generates schema
5. Storage engine:
   - Creates table/collection
   - Inserts data
6. Metadata indexer:
   - Stores metadata in `json_index` table
7. Returns confirmation with schema info

## Technology Stack

- **Frontend**: Next.js 14, React, TypeScript, Tailwind CSS
- **Backend**: FastAPI, Python 3.9+
- **ML/AI**: CLIP (OpenAI), Transformers, PyTorch
- **Databases**: PostgreSQL, MongoDB
- **File Storage**: Local filesystem (extensible to S3/MinIO)

## Edge Cases Handled

1. **Noisy JSON**: Defaults to NoSQL if consistency < 70%
2. **Unclassifiable Media**: Creates "uncategorized" category
3. **Schema Conflicts**: Each upload creates new schema with timestamp
4. **Large Files**: Handles via streaming (extensible for async processing)
5. **Missing Dependencies**: Graceful fallbacks for CLIP model

## Future Extensions

- Multilingual tagging
- User access control
- Advanced search UI with semantic search
- Versioned schemas and diffing
- S3/MinIO integration
- Video frame extraction
- Real-time processing status
- API rate limiting
- Authentication & authorization

## Getting Started

See [SETUP.md](SETUP.md) for detailed setup instructions.

Quick start:
```bash
# Start databases
docker-compose up -d

# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

Visit http://localhost:3000 to use the system.

