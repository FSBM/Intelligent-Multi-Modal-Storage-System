'use client';

import { useState, useCallback } from 'react';
import axios from 'axios';

interface UploadResult {
  status: string;
  type: string;
  filename: string;
  category?: string;
  storage_path?: string;
  schema_decision?: any;
  index_id?: number;
  error?: string;
  text_preview?: string;
  metadata?: any;
}

export default function FileUpload() {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [results, setResults] = useState<UploadResult[]>([]);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const newFiles = Array.from(e.dataTransfer.files);
      setFiles((prev) => [...prev, ...newFiles]);
    }
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files);
      setFiles((prev) => [...prev, ...newFiles]);
    }
  };

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const uploadFiles = async () => {
    if (files.length === 0) return;

    setUploading(true);
    setResults([]);

    const uploadResults: UploadResult[] = [];

    for (const file of files) {
      try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await axios.post('/api/upload', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });

        uploadResults.push(response.data);
      } catch (error: any) {
        uploadResults.push({
          status: 'error',
          type: 'unknown',
          filename: file.name,
          error: error.response?.data?.detail || error.message,
        });
      }
    }

    setResults(uploadResults);
    setUploading(false);
    setFiles([]);
  };

  return (
    <div className="space-y-6">
      {/* Upload Area */}
      <div
        className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
          dragActive
            ? 'border-primary-500 bg-primary-50'
            : 'border-gray-300 hover:border-gray-400'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          type="file"
          id="file-upload"
          className="hidden"
          multiple
          onChange={handleFileSelect}
          accept="image/*,video/*,.json,.pdf,.doc,.docx,.txt"
        />
        <label
          htmlFor="file-upload"
          className="cursor-pointer flex flex-col items-center"
        >
          <svg
            className="w-12 h-12 text-gray-400 mb-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
            />
          </svg>
          <p className="text-lg font-medium text-gray-700 mb-2">
            Drag and drop files here, or click to select
          </p>
          <p className="text-sm text-gray-500">
            Supports images, videos, JSON, PDF, DOC, DOCX, and TXT files
          </p>
        </label>
      </div>

      {/* Selected Files */}
      {files.length > 0 && (
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="font-medium text-gray-900 mb-3">Selected Files</h3>
          <div className="space-y-2">
            {files.map((file, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-2 bg-gray-50 rounded"
              >
                <span className="text-sm text-gray-700">{file.name}</span>
                <button
                  onClick={() => removeFile(index)}
                  className="text-red-500 hover:text-red-700 text-sm"
                >
                  Remove
                </button>
              </div>
            ))}
          </div>
          <button
            onClick={uploadFiles}
            disabled={uploading}
            className="mt-4 w-full bg-primary-600 text-white py-2 px-4 rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {uploading ? 'Uploading...' : `Upload ${files.length} file(s)`}
          </button>
        </div>
      )}

      {/* Upload Results */}
      {results.length > 0 && (
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="font-medium text-gray-900 mb-3">Upload Results</h3>
          <div className="space-y-3">
            {results.map((result, index) => (
              <div
                key={index}
                className={`p-4 rounded-lg ${
                  result.status === 'success'
                    ? 'bg-green-50 border border-green-200'
                    : 'bg-red-50 border border-red-200'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <p className="font-medium text-gray-900">{result.filename}</p>
                    {result.status === 'success' && (
                      <div className="mt-2 text-sm text-gray-600">
                        {result.type === 'media' && (
                          <>
                            <p>Type: Media</p>
                            {result.category && <p>Category: {result.category}</p>}
                            {result.storage_path && (
                              <p className="text-xs text-gray-500">
                                Path: {result.storage_path}
                              </p>
                            )}
                          </>
                        )}
                        {result.type === 'json' && (
                          <>
                            <p>Type: JSON</p>
                            {result.schema_decision && (
                              <p>
                                Storage: {result.schema_decision.storage_type}
                              </p>
                            )}
                          </>
                        )}
                        {result.type === 'document' && (
                          <>
                            <p>Type: Document</p>
                            {result.category && <p>Category: {result.category}</p>}
                            {result.storage_path && (
                              <p className="text-xs text-gray-500">
                                Path: {result.storage_path}
                              </p>
                            )}
                            {result.text_preview && (
                              <div className="mt-2 p-2 bg-gray-100 rounded text-xs max-h-32 overflow-y-auto">
                                <p className="font-medium mb-1">Text Preview:</p>
                                <p className="text-gray-700">{result.text_preview}...</p>
                              </div>
                            )}
                            {result.metadata?.metadata && (
                              <div className="mt-2 text-xs text-gray-500">
                                {result.metadata.metadata.word_count && (
                                  <p>Words: {result.metadata.metadata.word_count}</p>
                                )}
                                {result.metadata.metadata.num_pages && (
                                  <p>Pages: {result.metadata.metadata.num_pages}</p>
                                )}
                              </div>
                            )}
                          </>
                        )}
                      </div>
                    )}
                    {result.error && (
                      <p className="mt-2 text-sm text-red-600">{result.error}</p>
                    )}
                  </div>
                  <span
                    className={`px-2 py-1 rounded text-xs font-medium ${
                      result.status === 'success'
                        ? 'bg-green-100 text-green-800'
                        : 'bg-red-100 text-red-800'
                    }`}
                  >
                    {result.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

