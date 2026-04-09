# Gen Digital RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about Gen Digital using company materials. Built with Streamlit, LangChain, ChromaDB, and Google Gemini.

## Prerequisites

- **Python 3.10+**
- **Google API Key** — get one for free at [Google AI Studio](https://aistudio.google.com/apikey)

## Setup

1. **Clone the repository** and navigate into the project folder:

   ```bash
   cd gen-digital-rag-chatbot
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set your Google API key.** Choose one option:

   - **Option A — environment variable:**
     ```bash
     export GOOGLE_API_KEY="your-api-key-here"
     ```

   - **Option B — `.env` file:**  
     Create a file named `.env` in the project root with:
     ```
     GOOGLE_API_KEY=your-api-key-here
     ```

## Running the Chatbot

### Step 1: Ingest the data (run once)

This reads the company text file, splits it into chunks, generates embeddings, and stores them in a local ChromaDB vector database.

```bash
python ingest.py
```

You should see output like:
```
Loading data from data/gen_digital.txt ...
Split into XX chunks.
Creating vector store in ChromaDB ...
Done! Stored XX chunks in chroma_db
```

### Step 2: Start the chatbot

```bash
streamlit run app.py
```

The chatbot will open in your browser at **http://localhost:8501**.

Type any question about Gen Digital — products, brands, AI initiatives, culture, careers — and get answers grounded in the company materials.

## Project Structure

```
├── app.py                  # Streamlit chatbot application
├── ingest.py               # Data ingestion script (run once)
├── data/
│   └── gen_digital.txt     # Company knowledge base (plain text)
├── chroma_db/              # Vector database (auto-generated)
├── requirements.txt        # Python dependencies
├── .env                    # API key (create this yourself)
└── README.md
```

## Troubleshooting

- **"Run `python ingest.py` first"** — You need to run the ingestion step before starting the chatbot.
- **429 / quota errors** — The free Gemini API has rate limits. Wait a moment and try again, or generate a new API key.
- **No `GOOGLE_API_KEY`** — Make sure the key is set as an environment variable or in a `.env` file.
