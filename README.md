# CS 5542 — Lab 4

RAG (Retrieval-Augmented Generation) app with a Streamlit UI and optional FastAPI backend.

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd CS_5542_Lab_4
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

### 3. Activate the virtual environment

**macOS / Linux:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the app

```bash
streamlit run ui/app.py
```

---

## Project structure

| Path           | Description                    |
|----------------|--------------------------------|
| `backend/`     | RAG pipeline and API logic     |
| `ui/`          | Streamlit frontend (`app.py`)  |
| `data/`        | Documents, images, and logs    |
| `data/data/docs/`   | Document storage (add your files here) |
| `data/data/images/` | Image assets                   |
| `data/logs/`   | Logs (e.g. query metrics)      |
| `requirements.txt` | Python dependencies        |

---

## Notes

- **Logs:** The `data/logs/` folder is kept in the repo via `.gitkeep`. Generated files like `query_metrics.csv` are ignored (see `.gitignore`).
- **Data:** Put your RAG documents in `data/data/docs/` and any images in `data/data/images/`.
