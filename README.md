# Unified Smart Storage System

A unified storage system that intelligently ingests, analyzes, categorizes, stores, and retrieves any type of data: media (images/videos) and structured JSON.

## Architecture

```
┌─────────────────┐
│  Next.js Frontend │
└────────┬─────────┘
         │
┌────────▼─────────┐
│  Ingestion Gateway│ (FastAPI)
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼────┐
│ Media │ │  JSON │
│Processor│ │Analyzer│
└───┬───┘ └──┬────┘
    │        │
┌───▼────────▼───┐
│ Schema Builder │
└───┬────────────┘
    │
┌───▼──────────────┐
│ Storage Decision │
│     Engine       │
└───┬──────────────┘
    │
┌───▼──────────────┐
│  Directory Mgr   │
│  Metadata Index  │
└──────────────────┘
```

## Components

### Frontend (Next.js)
- Single interface for uploading media or JSON
- REST API integration
- Search and retrieval UI

### Ingestion Gateway (FastAPI)
- Detects data type (media vs JSON)
- Routes to correct processor
- Handles uploads and validation

### Media Processor
- Generates embeddings via CLIP/ViT
- Performs semantic classification
- Produces metadata

### JSON Analyzer
- Finds structure and patterns
- Handles batch JSON by merging structures
- Detects repeated fields

### Schema Builder
- Decides SQL (relational) or NoSQL (flexible)
- Auto-creates DB schema/entities

### Storage Decision Engine
- Chooses between SQL, NoSQL, or file storage
- Triggers provisioning of tables/collections/buckets

### Directory Manager
- Maintains folders/buckets for media categories
- Creates new directories or appends to existing ones

### Metadata + Indexing Layer
- Fast lookup for all stored files/records
- Enables efficient retrieval

### Monitoring & Logging
- Tracks ingestion, schema decisions, errors, and storage ops

## Tech Stack

- **Frontend**: Next.js / React
- **Gateway**: FastAPI
- **Media Processing**: Python + CLIP/ViT embeddings
- **JSON Analysis**: Python
- **SQL**: PostgreSQL
- **NoSQL**: MongoDB
- **File Storage**: Local FS or S3/MinIO

## Quick Start

### Using Docker Compose (Recommended)

1. **Start databases**
```bash
docker-compose up -d
```

2. **Backend Setup**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload
```

3. **Frontend Setup**
```bash
cd frontend
npm install
npm run dev
```

### Manual Setup

See [SETUP.md](SETUP.md) for detailed instructions.

### Prerequisites
- Python 3.9+
- Node.js 18+
- PostgreSQL (or use Docker Compose)
- MongoDB (or use Docker Compose)
- Docker & Docker Compose (optional, for databases)

## Usage

1. Upload media files (images/videos) through the web UI
2. Upload JSON objects (single or batch)
3. System automatically:
   - Detects data type
   - Processes and categorizes
   - Chooses optimal storage
   - Creates schemas if needed
   - Stores with metadata indexing

## Retrieval

- Media: Search by category, embeddings, metadata
- JSON: Query via DB with schema-aligned indexes

## Future Extensions

- Multilingual tagging
- User access control
- Advanced search UI
- Versioned schemas and diffing

