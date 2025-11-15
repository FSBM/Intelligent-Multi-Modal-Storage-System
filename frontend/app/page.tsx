'use client';

import { useState } from 'react';
import FileUpload from '@/components/FileUpload';
import SearchInterface from '@/components/SearchInterface';
import StatsDashboard from '@/components/StatsDashboard';

export default function Home() {
  const [activeTab, setActiveTab] = useState<'upload' | 'search' | 'stats'>('upload');

  return (
    <main className="min-h-screen p-8">
      <div className="max-w-7xl mx-auto">
        <header className="mb-8">
<<<<<<< HEAD
          <h1 className="text-4xl font-bold text-white-900 mb-2">
            Unified Smart Storage System
          </h1>
          <p className="text-white-600">
=======
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Unified Smart Storage System
          </h1>
          <p className="text-gray-600">
>>>>>>> 7317999ac0186241bdad8633188701b09657ab9f
            Intelligently store and retrieve media files and JSON data
          </p>
        </header>

        {/* Navigation Tabs */}
        <div className="mb-6 border-b border-gray-200">
          <nav className="flex space-x-8">
            <button
              onClick={() => setActiveTab('upload')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'upload'
                  ? 'border-primary-500 text-primary-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Upload
            </button>
            <button
              onClick={() => setActiveTab('search')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'search'
                  ? 'border-primary-500 text-primary-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Search
            </button>
            <button
              onClick={() => setActiveTab('stats')}
              className={`py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'stats'
                  ? 'border-primary-500 text-primary-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Statistics
            </button>
          </nav>
        </div>

        {/* Tab Content */}
        <div className="mt-6">
          {activeTab === 'upload' && <FileUpload />}
          {activeTab === 'search' && <SearchInterface />}
          {activeTab === 'stats' && <StatsDashboard />}
        </div>
      </div>
    </main>
  );
}

