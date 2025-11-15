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
  const [searchType, setSearchType] = useState<'media' | 'json'>('media');
  const [category, setCategory] = useState('');
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);

<<<<<<< HEAD
  // Search functions
=======
>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
  const searchMedia = async () => {
    setLoading(true);
    try {
      const response = await axios.get('/api/search/media', {
<<<<<<< HEAD
        params: { category: category || undefined, query: query || undefined, limit: 20 },
=======
        params: {
          category: category || undefined,
          query: query || undefined,
          limit: 20,
        },
>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
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
<<<<<<< HEAD
        params: { schema: category || undefined, query: query || undefined, limit: 20 },
=======
        params: {
          schema: category || undefined,
          query: query || undefined,
          limit: 20,
        },
>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
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
<<<<<<< HEAD
    searchType === 'media' ? searchMedia() : searchJSON();
=======
    if (searchType === 'media') {
      searchMedia();
    } else {
      searchJSON();
    }
>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
  };

  return (
    <div className="space-y-6">
<<<<<<< HEAD

      {/* Search Form */}
      <div className="bg-white rounded-lg shadow p-6">
        <form onSubmit={handleSearch} className="space-y-4">

          {/* Search Type */}
          <div>
            <label className="block text-sm font-medium text-black mb-2">Search Type</label>
            <div className="flex space-x-4">
              <label className="flex items-center text-black">
=======
      {/* Search Form */}
      <div className="bg-white rounded-lg shadow p-6">
        <form onSubmit={handleSearch} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Search Type
            </label>
            <div className="flex space-x-4">
              <label className="flex items-center">
>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
                <input
                  type="radio"
                  value="media"
                  checked={searchType === 'media'}
                  onChange={(e) => setSearchType(e.target.value as 'media')}
                  className="mr-2"
                />
                Media
              </label>
<<<<<<< HEAD
              <label className="flex items-center text-black">
=======
              <label className="flex items-center">
>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
                <input
                  type="radio"
                  value="json"
                  checked={searchType === 'json'}
                  onChange={(e) => setSearchType(e.target.value as 'json')}
                  className="mr-2"
                />
                JSON
              </label>
            </div>
          </div>

<<<<<<< HEAD
          {/* Category / Schema */}
          <div>
            <label className="block text-sm font-medium text-black mb-2">
=======
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
              {searchType === 'media' ? 'Category' : 'Schema'}
            </label>
            <input
              type="text"
              value={category}
              onChange={(e) => setCategory(e.target.value)}
<<<<<<< HEAD
              placeholder={searchType === 'media' ? 'e.g., nature, animals, people' : 'Schema name'}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500 text-black"
            />
          </div>

          {/* Query */}
          <div>
            <label className="block text-sm font-medium text-black mb-2">Query</label>
=======
              placeholder={
                searchType === 'media'
                  ? 'e.g., nature, animals, people'
                  : 'Schema name'
              }
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-primary-500 focus:border-primary-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Query
            </label>
>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search query"
<<<<<<< HEAD
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500 text-black"
=======
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-primary-500 focus:border-primary-500"
>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
            />
          </div>

          <button
            type="submit"
            disabled={loading}
<<<<<<< HEAD
            className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 disabled:opacity-50"
=======
            className="w-full bg-primary-600 text-white py-2 px-4 rounded-lg hover:bg-primary-700 disabled:opacity-50"
>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
        </form>
      </div>

<<<<<<< HEAD
      {/* Results */}
      {results.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="font-medium text-black mb-4">Results ({results.length})</h3>
=======
      {/* Search Results */}
      {results.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="font-medium text-gray-900 mb-4">
            Results ({results.length})
          </h3>
>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
          <div className="space-y-3">
            {results.map((result) => (
              <div
                key={result.id}
                className="p-4 border border-gray-200 rounded-lg hover:bg-gray-50"
              >
<<<<<<< HEAD
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
=======
                <p className="font-medium text-gray-900">{result.filename}</p>
                {searchType === 'media' && (
                  <div className="mt-2 text-sm text-gray-600">
                    {result.category && <p>Category: {result.category}</p>}
                    {result.mime_type && <p>Type: {result.mime_type}</p>}
                    {result.storage_path && (
                      <p className="text-xs text-gray-500">
                        Path: {result.storage_path}
                      </p>
                    )}
                  </div>
                )}
                {searchType === 'json' && (
                  <div className="mt-2 text-sm text-gray-600">
                    {result.schema_name && (
                      <p>Schema: {result.schema_name}</p>
                    )}
                    {result.storage_type && (
                      <p>Storage: {result.storage_type}</p>
                    )}
>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

<<<<<<< HEAD
      {/* No results */}
      {results.length === 0 && !loading && (
        <div className="bg-white rounded-lg shadow p-6 text-center text-black">
          No results found. Try adjusting your search criteria.
        </div>
      )}

    </div>
  );
}
=======
      {results.length === 0 && !loading && (
        <div className="bg-white rounded-lg shadow p-6 text-center text-gray-500">
          No results found. Try adjusting your search criteria.
        </div>
      )}
    </div>
  );
}

>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
