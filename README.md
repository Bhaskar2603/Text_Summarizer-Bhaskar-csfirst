# Text Summarizer

A web application that summarizes text documents using natural language processing.

## Features

- Text summarization
- File upload support (TXT, PDF, DOCX)
- Voice input with automatic punctuation
- Export summaries as DOCX
- Responsive web interface

## Prerequisites

- Docker
- Docker Compose

## Getting Started

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd text-summarizer
   ```

2. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

3. **Access the application**
   Open your browser and go to `http://localhost:5000`

## Environment Variables

You can configure the application using the following environment variables:

- `FLASK_APP`: The Flask application (default: `app.py`)
- `FLASK_ENV`: The environment (default: `production`)
- `FLASK_RUN_HOST`: The host to bind to (default: `0.0.0.0`)
- `FLASK_RUN_PORT`: The port to bind to (default: `5000`)

## Volumes

The application uses the following volumes:

- `/app`: Application code
- `/usr/local/nltk_data`: NLTK data files

## Building the Docker Image

To build the Docker image manually:

```bash
docker build -t text-summarizer .
```

## Running the Container

To run the container manually:

```bash
docker run -d -p 5000:5000 --name text-summarizer text-summarizer
```

## Stopping the Application

To stop the application:

```bash
docker-compose down
```

## Development

For development, you can mount the source code as a volume:

```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

## License

This project is licensed under the MIT License.
