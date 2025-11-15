# Setup Instructions

## Prerequisites

1. **Python 3.9+**
   ```bash
   python --version
   ```

2. **Node.js 18+**
   ```bash
   node --version
   ```

3. **PostgreSQL**
   - Install PostgreSQL: https://www.postgresql.org/download/
   - Create database:
     ```sql
     CREATE DATABASE smart_storage;
     ```

4. **MongoDB**
   - Install MongoDB: https://www.mongodb.com/try/download/community
   - MongoDB will run on default port 27017

## Backend Setup

1. **Navigate to backend directory**
   ```bash
   cd backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your database credentials
   ```

5. **Start backend server**
   ```bash
   uvicorn main:app --reload
   ```
   Backend will run on http://localhost:8000

## Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start development server**
   ```bash
   npm run dev
   ```
   Frontend will run on http://localhost:3000

## Usage

1. Open http://localhost:3000 in your browser
2. Upload media files (images/videos) or JSON files
3. System will automatically:
   - Detect file type
   - Process and categorize
   - Choose optimal storage
   - Create schemas if needed
   - Store with metadata indexing

## Testing

### Test Media Upload
- Upload an image file (jpg, png, etc.)
- System will generate embeddings and classify into category
- File stored in `storage/media/{category}/`

### Test JSON Upload
- Upload a JSON file (single object or array)
- System will analyze structure
- Decide SQL vs NoSQL
- Auto-create schema and store data

## Troubleshooting

### Backend Issues
- Check database connections in `.env`
- Ensure PostgreSQL and MongoDB are running
- Check logs in console for errors

### Frontend Issues
- Clear browser cache
- Check browser console for errors
- Verify backend is running on port 8000


## Production Deployment

1. Set `LOG_LEVEL=WARNING` in production
2. Use proper database credentials
3. Configure S3/MinIO for file storage
4. Set up proper CORS origins
5. Use environment-specific configurations

