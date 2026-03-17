import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import axios from 'axios'

function DocumentUpload({ onSuccess, onError }) {
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)

  const onDrop = useCallback(async (acceptedFiles) => {
    const file = acceptedFiles[0]
    if (!file) return

    if (!file.name.endsWith('.pdf')) {
      onError('Only PDF files are supported')
      return
    }

    setUploading(true)
    setUploadProgress(0)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await axios.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const progress = Math.round(
            (progressEvent.loaded * 100) / (progressEvent.total || 1)
          )
          setUploadProgress(progress)
        },
      })

      onSuccess(response.data)
    } catch (err) {
      const message = err.response?.data?.error || 'Upload failed'
      onError(message)
    } finally {
      setUploading(false)
      setUploadProgress(0)
    }
  }, [onSuccess, onError])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
    },
    multiple: false,
    disabled: uploading,
  })

  return (
    <div
      {...getRootProps()}
      className={`upload-area ${isDragActive ? 'drag-active' : ''}`}
    >
      <input {...getInputProps()} />

      {uploading ? (
        <div className="loading">
          <div className="spinner" />
          <span>Uploading... {uploadProgress}%</span>
        </div>
      ) : (
        <>
          <div className="upload-icon">📄</div>
          <p className="upload-text">
            {isDragActive
              ? 'Drop the PDF here'
              : 'Drag & drop a PDF, or click to select'}
          </p>
          <p className="upload-hint">Maximum file size: 16MB</p>
        </>
      )}
    </div>
  )
}

export default DocumentUpload
