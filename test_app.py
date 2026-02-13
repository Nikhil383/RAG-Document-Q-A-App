import pytest
from app import app
import io

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    rv = client.get('/')
    assert rv.status_code == 200
    assert b'Agentic RAG with Gemini' in rv.data

def test_upload_no_file(client):
    rv = client.post('/upload')
    assert rv.status_code == 400
    assert b'No file part' in rv.data

def test_chat_no_context(client):
    rv = client.post('/chat', json={'question': 'Hello'})
    assert rv.status_code == 400
    assert b'Please upload a PDF first' in rv.data
