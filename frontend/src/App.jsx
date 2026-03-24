import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Upload, Database, Send, Network, Loader2, Link2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import './index.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setIsUploading(true);
    setUploadStatus('Graph Extraction Active... (this takes a moment)');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setUploadStatus(response.data.message || 'Successfully ingested to Neo4j.');
    } catch (err) {
      setUploadStatus(`Error: ${err.response?.data?.error || err.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handleChat = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const query = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: query }]);
    setLoading(true);

    try {
      const response = await axios.post('/api/chat', { query });
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: response.data.answer,
        entities: response.data.entities_searched,
        context: response.data.context_used
      }]);
    } catch (err) {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: `Error: ${err.response?.data?.error || err.message}` 
      }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar / Setup Panel */}
      <aside className="sidebar">
        <h2>
          <Network />
          Graph RAG
        </h2>
        <p style={{ color: 'var(--text-dim)', fontSize: '0.85rem', marginBottom: '1.5rem', lineHeight: 1.5 }}>
          Upload a document to extract Entities and Relationships into Neo4j using Gemini.
        </p>

        <div 
          className="upload-box" 
          onClick={() => !isUploading && fileInputRef.current.click()}
          style={{ opacity: isUploading ? 0.5 : 1, cursor: isUploading ? 'not-allowed' : 'pointer' }}
        >
          {isUploading ? (
            <Loader2 className="lucide-spin" size={32} style={{ color: 'var(--primary)', margin: '0 auto 1rem' }} />
          ) : (
            <Upload size={32} style={{ color: 'var(--primary)', margin: '0 auto 1rem' }} />
          )}
          
          <div style={{ fontWeight: 600 }}>{isUploading ? 'Extracting Nodes...' : 'Upload Document'}</div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginTop: '0.5rem' }}>
            .txt, .md, .pdf
          </div>
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileUpload} 
            className="upload-input"
            accept=".txt,.md,.pdf"
          />
        </div>

        {uploadStatus && (
          <div className="status-text">{uploadStatus}</div>
        )}

        <div style={{ marginTop: 'auto', paddingTop: '1rem', borderTop: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-dim)', fontSize: '0.8rem' }}>
          <Database size={16} /> Connected to Neo4j
        </div>
      </aside>

      {/* Main Chat Interface */}
      <main className="main-content">
        <div className="chat-header">
          LangGraph Agent Execution
        </div>

        <div className="chat-messages">
          {messages.length === 0 ? (
            <div style={{ margin: 'auto', textAlign: 'center', color: 'var(--text-dim)' }}>
              Ask a question about the Knowledge Graph.
            </div>
          ) : (
            <AnimatePresence>
              {messages.map((msg, idx) => (
                <motion.div 
                  key={idx}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`message ${msg.role === 'user' ? 'user' : 'bot'}`}
                >
                  <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                  
                  {/* Graph RAG Metadata attached to bot responses */}
                  {msg.role === 'assistant' && msg.entities && msg.entities.length > 0 && (
                    <div style={{ marginTop: '1rem', paddingTop: '0.5rem', borderTop: '1px solid var(--border)' }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginBottom: '4px' }}>
                        Entities Extracted:
                      </div>
                      <div className="entity-tags">
                        {msg.entities.map((ent, i) => (
                          <span key={i} className="entity-tag">{ent}</span>
                        ))}
                      </div>
                      
                      {msg.context && msg.context !== "No specific relationships found in the knowledge graph." && (
                        <details style={{ marginTop: '0.5rem', fontSize: '0.75rem' }}>
                          <summary style={{ cursor: 'pointer', color: 'var(--text-dim)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                            <Link2 size={12} /> Cypher Graph Context
                          </summary>
                          <div style={{ marginTop: '0.5rem', padding: '0.5rem', background: 'rgba(0,0,0,0.2)', borderRadius: '4px', whiteSpace: 'pre-wrap', fontFamily: 'monospace' }}>
                            {msg.context}
                          </div>
                        </details>
                      )}
                    </div>
                  )}
                </motion.div>
              ))}
            </AnimatePresence>
          )}
          {loading && (
            <div className="message bot" style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
              <Loader2 className="lucide-spin" size={16} /> Agent traversing graph...
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="chat-input">
          <form onSubmit={handleChat} className="chat-form">
            <input 
              type="text" 
              value={input} 
              onChange={(e) => setInput(e.target.value)} 
              placeholder="Query the Agent..."
              disabled={loading}
            />
            <button type="submit" disabled={!input.trim() || loading}>
              <Send size={18} />
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}

export default App;
