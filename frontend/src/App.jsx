import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import DocumentUpload from './components/DocumentUpload'
import DocumentList from './components/DocumentList'
import Chat from './components/Chat'
import './App.css'

function App() {
  const [documents, setDocuments] = useState([])
  const [sessionInfo, setSessionInfo] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Fetch session info and documents on mount
  useEffect(() => {
    fetchSessionInfo()
    fetchDocuments()
  }, [])

  const fetchSessionInfo = async () => {
    try {
      const response = await axios.get('/api/session')
      setSessionInfo(response.data)
    } catch (err) {
      console.error('Failed to fetch session info:', err)
    }
  }

  const fetchDocuments = async () => {
    try {
      const response = await axios.get('/api/documents')
      setDocuments(response.data.documents || [])
    } catch (err) {
      console.error('Failed to fetch documents:', err)
    }
  }

  const handleUploadSuccess = useCallback((result) => {
    setDocuments(prev => [...prev, {
      id: result.document_id,
      name: result.document_name,
      pages: result.pages,
      chunks: result.chunks
    }])
    setError(null)
    fetchSessionInfo()
  }, [])

  const handleUploadError = useCallback((errorMessage) => {
    setError(errorMessage)
  }, [])

  const handleDeleteDocument = async (docId) => {
    try {
      await axios.delete(`/api/documents/${docId}`)
      setDocuments(prev => prev.filter(doc => doc.id !== docId))
      fetchSessionInfo()
    } catch (err) {
      setError('Failed to delete document')
    }
  }

  const handleClearAll = async () => {
    if (!confirm('Are you sure you want to remove all documents?')) return

    try {
      await axios.post('/api/documents/clear')
      setDocuments([])
      fetchSessionInfo()
    } catch (err) {
      setError('Failed to clear documents')
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Agentic RAG with Gemini</h1>
        <p>Upload PDFs and ask questions about their content</p>
      </header>

      <main className="main">
        <aside className="sidebar">
          <div className="card">
            <h2>Upload Document</h2>
            <DocumentUpload
              onSuccess={handleUploadSuccess}
              onError={handleUploadError}
            />
            {error && (
              <div className="alert alert-error" style={{ marginTop: '1rem' }}>
                {error}
              </div>
            )}
          </div>

          <div className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h2>Documents</h2>
              {documents.length > 0 && (
                <button
                  className="btn btn-secondary"
                  style={{ padding: '0.5rem 1rem', fontSize: '0.85rem' }}
                  onClick={handleClearAll}
                >
                  Clear All
                </button>
              )}
            </div>
            <DocumentList
              documents={documents}
              onDelete={handleDeleteDocument}
            />
            {sessionInfo && (
              <div className="session-info">
                Session: {sessionInfo.session_id?.slice(0, 8)}... • {documents.length} doc{documents.length !== 1 ? 's' : ''}
              </div>
            )}
          </div>
        </aside>

        <section className="chat-section">
          <Chat documents={documents} />
        </section>
      </main>
    </div>
  )
}

export default App
