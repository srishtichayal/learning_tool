# PDF to Interactive Presentations

## Overview

This project provides an interface to convert PDF documents into interactive, learnable presentations. Users can summarize PDFs page by page and generate slides or video presentations, optionally with narration. The focus is on creating content that is engaging and easy to understand for students.

### Approach

- **Streamlit Interface:** Provides a user-friendly interface to upload PDFs, preview summaries, and generate slides or videos.
- **Summarization Options:**
  - **OpenAI GPT-4:** High-quality summarization with prompt engineering for engaging and learnable content. Requires a valid OpenAI API key.
  - **T5-small (Hugging Face):** Works offline on CPU; quality is reasonable but not as good as OpenAI.
- **Page-wise Summarization:** Each PDF page is summarized individually.
- **Preview Feature:** Users can preview summaries before generating slides or videos.
- **Output Options:** 
  - Slides with narration (video)
  - Slides without narration (PPTX)

---

## Tech Stack & Models Used

- **Python** for backend logic
- **Hugging Face Transformers** (T5-small)
- **OpenAI GPT-4** for optional high-quality summarization
- **Python-pptx** for slide generation
- **pyttsx3** for audio narration
- **MoviePy** for video creation
- **Streamlit** for web interface
- **Docker** for containerized deployment

---

## Limitations & Possible Improvements

- **Improved slide design and layout options:** Could support multiple templates for better visual appeal.
- **Enhanced Narration:** Could integrate AI-generated speakers with customizable faces, accents, and tones for higher engagement.
- **Language Support:** Currently limited due to resource constraints; more languages can be added.
- **Advanced Features:** Tone and style customization for narration could be added. Bigger PDFs could be supported.

---

## Usage

### Running Locally

1. **Clone the repository**:

```bash
git clone <your-repo-url>
cd <repo-folder>
```

2. **Install dependencies (if not using Docker), I suggest making a virtual environment:**:

```bash
pip install -r requirements.txt
```

3. **Run the app:**:

```bash
streamlit run streamlit_app.py
```

### Running with Docker

1. **Build Docker image:**:

```bash
docker build -t pdf-summarizer-app
```

2. **Run the app in a container:**:

```bash
docker run -p 8501:8501 pdf-summarizer-app
```

### Notes

- OpenAI API key is optional but recommended for better summaries.\n
- Preview summaries before generating outputs to make adjustments if needed.\n
- Both slides and videos can be downloaded for offline use.


