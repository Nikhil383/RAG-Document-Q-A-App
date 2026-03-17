function DocumentList({ documents, onDelete }) {
  if (documents.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">📂</div>
        <p>No documents uploaded yet</p>
        <p style={{ fontSize: '0.85rem', marginTop: '0.5rem' }}>
          Upload a PDF to get started
        </p>
      </div>
    )
  }

  return (
    <div className="document-list">
      {documents.map((doc) => (
        <div key={doc.id} className="document-item">
          <div className="document-info">
            <div className="document-name" title={doc.name}>
              📄 {doc.name}
            </div>
            <div className="document-meta">
              {doc.pages && `${doc.pages} pages`}
              {doc.chunks && ` • ${doc.chunks} chunks`}
            </div>
          </div>
          <div className="document-actions">
            <button
              className="btn-icon"
              onClick={() => onDelete(doc.id)}
              title="Delete document"
            >
              🗑️
            </button>
          </div>
        </div>
      ))}
    </div>
  )
}

export default DocumentList
