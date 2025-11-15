'use client';

import { useState, useEffect } from 'react';
import axios from 'axios';

interface Stats {
  media_files: number;
  json_files: number;
  categories: number;
  schemas: number;
  category_list: string[];
  schema_list: string[];
}

export default function StatsDashboard() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await axios.get('/api/stats');
      setStats(response.data);
    } catch (error) {
      console.error('Error fetching stats:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow p-6 text-center">
        <p className="text-gray-500">Loading statistics...</p>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="bg-white rounded-lg shadow p-6 text-center">
        <p className="text-red-500">Failed to load statistics</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-sm font-medium text-gray-500 mb-2">Media Files</h3>
          <p className="text-3xl font-bold text-gray-900">{stats.media_files}</p>
        </div>
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-sm font-medium text-gray-500 mb-2">JSON Files</h3>
          <p className="text-3xl font-bold text-gray-900">{stats.json_files}</p>
        </div>
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-sm font-medium text-gray-500 mb-2">Categories</h3>
          <p className="text-3xl font-bold text-gray-900">{stats.categories}</p>
        </div>
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-sm font-medium text-gray-500 mb-2">Schemas</h3>
          <p className="text-3xl font-bold text-gray-900">{stats.schemas}</p>
        </div>
      </div>

      {/* Categories List */}
      {stats.category_list.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="font-medium text-gray-900 mb-4">Media Categories</h3>
          <div className="flex flex-wrap gap-2">
            {stats.category_list.map((category) => (
              <span
                key={category}
                className="px-3 py-1 bg-primary-100 text-primary-800 rounded-full text-sm"
              >
                {category}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Schemas List */}
      {stats.schema_list.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="font-medium text-gray-900 mb-4">JSON Schemas</h3>
          <div className="flex flex-wrap gap-2">
            {stats.schema_list.map((schema) => (
              <span
                key={schema}
                className="px-3 py-1 bg-gray-100 text-gray-800 rounded-full text-sm"
              >
                {schema}
              </span>
            ))}
          </div>
        </div>
      )}

      {stats.category_list.length === 0 && stats.schema_list.length === 0 && (
        <div className="bg-white rounded-lg shadow p-6 text-center text-gray-500">
          No data stored yet. Upload some files to see statistics.
        </div>
      )}
    </div>
  );
}

