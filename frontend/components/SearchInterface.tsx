'use client';

import { useState } from 'react';
import axios from 'axios';

interface SearchResult {
  id: number;
  filename: string;
  category?: string;
  mime_type?: string;
  storage_path?: string;
  schema_name?: string;
  storage_type?: string;
  [key: string]: any;
}

export default function SearchInterface() {
  const [searchType, setSearchType] = useState<'media' | 'json' | 'documents'>('media');
  const [category, setCategory] = useState('');
  const [query, setQuery] = useState('');
  const [mimeType, setMimeType] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);

  // Search functions
  const searchMedia = async () => {
    setLoading(true);
    try {
      const response = await axios.get('/api/search/media', {
        params: { category: category || undefined, query: query || undefined, limit: 20 },
      });
      setResults(response.data.results || []);
    } catch (error) {
      console.error('Search error:', error);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const searchJSON = async () => {
    setLoading(true);
    try {
      const response = await axios.get('/api/search/json', {
        params: { schema: category || undefined, query: query || undefined, limit: 20 },
      });
      setResults(response.data.results || []);
    } catch (error) {
      console.error('Search error:', error);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const searchDocuments = async () => {
    setLoading(true);
    try {
      const response = await axios.get('/api/search/documents', {
        params: {
          category: category || undefined,
          mime_type: mimeType || undefined,
          query: query || undefined,
          limit: 20,
        },
      });
      setResults(response.data.results || []);
    } catch (error) {
      console.error('Search error:', error);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchType === 'media') {
      searchMedia();
    } else if (searchType === 'json') {
      searchJSON();
    } else {
      searchDocuments();
    }
  };

  return (
    <div className="space-y-6">

      {/* Search Form */}
      <div className="bg-white rounded-lg shadow p-6">
        <form onSubmit={handleSearch} className="space-y-4">

          {/* Search Type */}
          <div>
            <label className="block text-sm font-medium text-black mb-2">Search Type</label>
            <div className="flex space-x-4">
              <label className="flex items-center text-black">
                <input
                  type="radio"
                  value="media"
                  checked={searchType === 'media'}
                  onChange={(e) => setSearchType(e.target.value as 'media' | 'json' | 'documents')}
                  className="mr-2"
                />
                Media
              </label>
              <label className="flex items-center text-black">
                <input
                  type="radio"
                  value="json"
                  checked={searchType === 'json'}
                  onChange={(e) => setSearchType(e.target.value as 'media' | 'json' | 'documents')}
                  className="mr-2"
                />
                JSON
              </label>
              <label className="flex items-center text-black">
                <input
                  type="radio"
                  value="documents"
                  checked={searchType === 'documents'}
                  onChange={(e) => setSearchType(e.target.value as 'media' | 'json' | 'documents')}
                  className="mr-2"
                />
                Documents
              </label>
            </div>
          </div>

          {/* Category / Schema */}
          <div>
            <label className="block text-sm font-medium text-black mb-2">
              {searchType === 'media' ? 'Category' : searchType === 'json' ? 'Schema' : 'Category'}
            </label>
            <input
              type="text"
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              placeholder={
                searchType === 'media'
                  ? 'e.g., nature, animals, people'
                  : searchType === 'json'
                  ? 'Schema name'
                  : 'e.g., technical, business, academic'
              }
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500 text-black"
            />
          </div>

          {/* MIME Type (for documents) */}
          {searchType === 'documents' && (
            <div>
              <label className="block text-sm font-medium text-black mb-2">MIME Type (optional)</label>
              <select
                value={mimeType}
                onChange={(e) => setMimeType(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500 text-black"
              >
                <option value="">All types</option>
                <option value="application/pdf">PDF</option>
                <option value="application/msword">DOC</option>
                <option value="application/vnd.openxmlformats-officedocument.wordprocessingml.document">DOCX</option>
                <option value="text/plain">TXT</option>
              </select>
            </div>
          )}

          {/* Query */}
          <div>
            <label className="block text-sm font-medium text-black mb-2">Query</label>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search query"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500 text-black"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
        </form>
      </div>

      {/* Results */}
      {results.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="font-medium text-black mb-4">Results ({results.length})</h3>
          <div className="space-y-3">
            {results.map((result) => (
              <div
                key={result.id}
                className="p-4 border border-gray-200 rounded-lg hover:bg-gray-50"
              >
                <p className="font-medium text-black">{result.filename}</p>

                {searchType === 'media' && (
                  <div className="mt-2 text-sm text-black">
                    {result.category && <p>Category: {result.category}</p>}
                    {result.mime_type && <p>Type: {result.mime_type}</p>}
                    {result.storage_path && <p className="text-xs text-gray-700">Path: {result.storage_path}</p>}
                  </div>
                )}

                {searchType === 'json' && (
                  <div className="mt-2 text-sm text-black">
                    {result.schema_name && <p>Schema: {result.schema_name}</p>}
                    {result.storage_type && <p>Storage: {result.storage_type}</p>}
                  </div>
                )}

                {searchType === 'documents' && (
                  <div className="mt-2 text-sm text-black">
                    {result.category && <p>Category: {result.category}</p>}
                    {result.mime_type && <p>Type: {result.mime_type}</p>}
                    {result.storage_path && <p className="text-xs text-gray-700">Path: {result.storage_path}</p>}
                    {result.text && (
                      <p className="text-xs text-gray-600 mt-1 line-clamp-2">
                        {result.text.substring(0, 200)}...
                      </p>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* No results */}
      {results.length === 0 && !loading && (
        <div className="bg-white rounded-lg shadow p-6 text-center text-black">
          No results found. Try adjusting your search criteria.
        </div>
      )}

    </div>
  );
}
