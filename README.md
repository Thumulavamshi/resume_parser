# Resume Parser API

A FastAPI-powered resume parsing service that extracts structured information from PDF and text files using Groq AI.

## Features

- 📄 Parse PDF and TXT resume files
- 🤖 AI-powered extraction using Groq's Llama model
- 🔄 Structured JSON output with personal info and professional details
- 🔒 Input validation and security features
- 📚 Interactive API documentation
- 🌐 CORS enabled for web integration

## Quick Start

### Prerequisites

- Python 3.8+
- Groq API Key ([Get one here](https://groq.com/))

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd parser
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

4. Run the server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### `POST /parse-resume`
Upload and parse a resume file.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (PDF or TXT file, max 10MB)

**Response:**
```json
{
  "personal_info": {
    "name": "John Doe",
    "email": "john.doe@email.com",
    "phone": "123-456-7890",
    "linkedin": "linkedin.com/in/johndoe",
    "github": "github.com/johndoe",
    "website": "johndoe.com"
  },
  "other_info": {
    "education": [...],
    "experience": [...],
    "projects": [...],
    "extra_info": {
      "skills": [...],
      "achievements": [...],
      "certifications": [...]
    }
  }
}
```

### `GET /health`
Check API health and configuration status.

**Response:**
```json
{
  "status": "healthy",
  "groq_key_configured": true
}
```

### `GET /`
API information and status.

## Usage Examples

### Using curl
```bash
curl -X POST "http://localhost:8000/parse-resume" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@resume.pdf"
```

### Using Postman
1. Set method to `POST`
2. URL: `http://localhost:8000/parse-resume`
3. Body → form-data
4. Key: `file` (set type to File)
5. Value: Select your resume file

### Using Python requests
```python
import requests

url = "http://localhost:8000/parse-resume"
files = {"file": open("resume.pdf", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## Deployment

### Using Docker (Recommended)

1. Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

2. Build and run:
```bash
docker build -t resume-parser .
docker run -p 8000:8000 --env-file .env resume-parser
```

### Using Production Server

For production, use a production ASGI server like Gunicorn:

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Your Groq API key | Yes |
| `MODEL` | Groq model to use | No (default: llama-3.1-8b-instant) |
| `PORT` | Server port | No (default: 8000) |

## File Support

- **PDF files**: Extracted using PyPDF2
- **Text files**: Direct text processing
- **File size limit**: 10MB maximum
- **File extensions**: `.pdf`, `.txt`

## Error Handling

The API provides detailed error messages for common issues:

- `400`: Invalid file type or size
- `500`: GROQ_API_KEY not configured
- `500`: LLM processing errors

## Development

### Interactive API Docs
Visit `http://localhost:8000/docs` for Swagger UI documentation.

### Project Structure
```
parser/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── .env.example        # Environment template
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- Create an issue on GitHub
- Check the interactive docs at `/docs`
- Review the health endpoint at `/health`