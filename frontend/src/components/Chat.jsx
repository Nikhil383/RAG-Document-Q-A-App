import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'

function Chat({ documents }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [showSources, setShowSources] = useState({})
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!input.trim() || loading) return

    const question = input.trim()
    setInput('')

    // Add user message
    const userMessage = { id: Date.now(), role: 'user', content: question }
    setMessages(prev => [...prev, userMessage])

    setLoading(true)

    try {
      const response = await axios.post('/chat', {
        question,
        include_sources: true,
      })

      const assistantMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: response.data.answer,
        sources: response.data.sources,
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (err) {
      const errorCode = err.response?.data?.code
      const errorMessage = err.response?.data?.error || 'Failed to get response'

      const systemMessage = {
        id: Date.now() + 1,
        role: 'system',
        content: errorCode === 'NO_DOCUMENTS'
          ? 'Please upload a document first before asking questions.'
          : errorMessage,
      }

      setMessages(prev => [...prev, systemMessage])
    } finally {
      setLoading(false)
    }
  }

  const toggleSources = (messageId) => {
    setShowSources(prev => ({
      ...prev,
      [messageId]: !prev[messageId],
    }))
  }

  const hasDocuments = documents.length > 0

  return (
    <>
      <div className="chat-header">
        <h2>Chat</h2>
        {hasDocuments && (
          <span style={{ fontSize: '0.85rem', color: 'var(--success-color)' }}>
            ● Ready
          </span>
        )}
      </div>

      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="welcome-screen">
            <div className="welcome-icon">💬</div>
            <h3 className="welcome-title">Start a conversation</h3>
            <p className="welcome-text">
              {hasDocuments
                ? 'Ask questions about your uploaded documents. The AI will retrieve relevant information and provide cited answers.'
                : 'Upload a PDF document first, then ask questions about its content.'}
            </p>
          </div>
        ) : (
          <>
            {messages.map((message) => (
              <div key={message.id} className={`message ${message.role}`}>
                <div className="message-header">
                  {message.role === 'user' && 'You'}
                  {message.role === 'assistant' && 'Assistant'}
                  {message.role === 'system' && 'System'}
                </div>
                <div className="message-content">
                  <ReactMarkdown>{message.content}</ReactMarkdown>
                </div>

                {message.sources && message.sources.length > 0 && (
                  <div className="sources-toggle">
                    <button
                      className="sources-btn"
                      onClick={() => toggleSources(message.id)}
                    >
                      {showSources[message.id] ? '▼' : '▶'} Sources ({message.sources.length})
                    </button>

                    {showSources[message.id] && (
                      <div className="sources-list">
                        {message.sources.map((source, idx) => (
                          <div key={idx} className="source-item">
                            <div className="source-meta">
                              {source.metadata?.page_number && `Page ${source.metadata.page_number}`}
                              {source.metadata?.document_name && ` from ${source.metadata.document_name}`}
                              {source.distance && ` (score: ${(1 - source.distance).toFixed(3)})`}
                            </div>
                            <div className="source-text">{source.text}</div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}

            {loading && (
              <div className="message assistant">
                <div className="message-header">Assistant</div>
                <div className="typing-indicator">
                  <span />
                  <span />
                  <span />
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      <div className="chat-input-container">
        <form onSubmit={handleSubmit} className="chat-input-form">
          <input
            type="text"
            className="chat-input"
            placeholder={
              hasDocuments
                ? 'Ask a question about your documents...'
                : 'Upload a document first...'
            }
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={!hasDocuments || loading}
          />
          <button
            type="submit"
            className="btn"
            disabled={!hasDocuments || !input.trim() || loading}
          >
            Send
          </button>
        </form>
      </div>
    </>
  )
}

export default Chat
