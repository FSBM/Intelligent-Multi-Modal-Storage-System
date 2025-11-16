# Quick Testing Guide

## Quick Start for Testing

### 1. Start Backend (Terminal 1)
```bash
cd backend
python main.py
```
Backend should start on `http://localhost:8000`

### 2. Start Frontend (Terminal 2)
```bash
cd frontend
npm install  # First time only
npm run dev
```
Frontend should start on `http://localhost:3000`

### 3. Test in Browser
1. Open `http://localhost:3000`
2. Go to "Upload" tab
3. Upload test files:
   - A video file (`.mp4`, `.mov`, etc.)
   - A PDF file
   - A DOC/DOCX file
   - A TXT file

### 4. Verify Categories
After upload, check that:
- ✅ Videos show categories like "animals", "food", "vehicles", etc. (NOT "uncategorized")
- ✅ Documents show categories like "technical", "business", "academic", etc. (NOT "uncategorized")
- ✅ Text preview appears for documents
- ✅ Metadata shows correctly (word count, page count, etc.)

### 5. Test Search
1. Go to "Search" tab
2. Select "Documents" search type
3. Search by category or query
4. Verify results show correctly

## Expected Categories

### Videos
- Animal videos → "animals"
- Food videos → "food"
- Vehicle videos → "vehicles"
- Nature videos → "nature"
- People videos → "people"
- Sports videos → "sports"
- Technology videos → "technology"
- Architecture videos → "architecture"

### Documents
- Code/Technical docs → "technical"
- Research papers → "academic"
- Business docs → "business"
- Legal docs → "legal"
- Medical docs → "medical"
- Financial docs → "financial"
- Educational content → "educational"
- Literature → "literature"
- Scientific papers → "scientific"
- News articles → "news"

## Troubleshooting

### If categories show "uncategorized":

1. **For Videos:**
   - Install video processing: `pip install imageio imageio-ffmpeg opencv-python`
   - Restart backend
   - Try uploading again

2. **For Documents:**
   - Install document processing: `pip install PyPDF2 pdfplumber python-docx`
   - Restart backend
   - Try uploading again

3. **Check Backend Logs:**
   - Look for error messages in backend terminal
   - Check if libraries are installed correctly

## Test Files to Create

Create these test files to verify categorization:

1. **test_video_animals.mp4** - Any video with animals
2. **test_technical.pdf** - PDF with code/technical content
3. **test_business.docx** - Word doc with business content
4. **test_academic.txt** - Text file with research content

Upload these and verify they get categorized correctly!

