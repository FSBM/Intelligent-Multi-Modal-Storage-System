# Quick Start Guide - How to Run the Server

## Step-by-Step Instructions

### Step 1: Start Databases

**Option A: Using Docker (Recommended)**
```bash
# From project root
docker compose up -d
# OR if you have older Docker:
docker-compose up -d
```

This starts PostgreSQL on port 5432 and MongoDB on port 27017.

**Option B: Manual Setup**
- Install and start PostgreSQL
- Install and start MongoDB
- Create database: `CREATE DATABASE smart_storage;`

### Step 2: Set Up Backend Server

Open a **new terminal window** and run:

```bash
# Navigate to backend
cd backend

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env

# Start the backend server
uvicorn main:app --reload
```

✅ Backend will run on **http://localhost:8000**

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### Step 3: Set Up Frontend Server

Open **another new terminal window** and run:

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start the frontend server
npm run dev
```

✅ Frontend will run on **http://localhost:3000**

You should see:
```
✓ Ready in X seconds
○ Local: http://localhost:3000
```

### Step 4: Access the Application

1. Open your browser
2. Go to **http://localhost:3000**
3. You should see the "Unified Smart Storage System" interface

## Testing the Server

### Test Backend API
```bash
# Check if backend is running
curl http://localhost:8000/

# Should return: {"status":"ok","service":"Unified Smart Storage System"}
```

### Test Frontend
- Open http://localhost:3000
- You should see the upload interface

## Common Issues & Solutions

### Issue: "Port 8000 already in use"
**Solution:** Kill the process using port 8000:
```bash
# macOS/Linux
lsof -ti:8000 | xargs kill -9

# Or change port in uvicorn command:
uvicorn main:app --reload --port 8001
```

### Issue: "Port 3000 already in use"
**Solution:** Next.js will automatically use the next available port, or:
```bash
# Kill process on port 3000
lsof -ti:3000 | xargs kill -9
```

### Issue: Database connection errors
**Solution:** 
1. Check if databases are running:
   ```bash
   # Check PostgreSQL
   docker ps | grep postgres
   
   # Check MongoDB
   docker ps | grep mongo
   ```

2. Verify `.env` file has correct database URLs:
   ```
   DATABASE_URL=postgresql://postgres:postgres@localhost:5432/smart_storage
   MONGODB_URL=mongodb://localhost:27017/
   ```

### Issue: "Module not found" errors
**Solution:** Make sure virtual environment is activated and dependencies are installed:
```bash
cd backend
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: CLIP model download is slow
**Solution:** This is normal on first run. The CLIP model (~500MB) downloads automatically. Be patient or ensure stable internet connection.

## Running in Production

For production, use:

**Backend:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
npm run build
npm start
```

## Stopping the Servers

- **Backend:** Press `Ctrl+C` in the backend terminal
- **Frontend:** Press `Ctrl+C` in the frontend terminal
- **Databases:** `docker compose down` (if using Docker)

## Need Help?

- Check logs in the terminal windows
- Verify all prerequisites are installed
- See [SETUP.md](SETUP.md) for detailed setup instructions
- Check [README.md](README.md) for architecture overview

