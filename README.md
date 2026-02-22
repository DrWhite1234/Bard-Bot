# BardBot: A RAG Chatbot on Shakespeare’s Works

## Author : Aswath S

**BardBot** is a Retrieval-Augmented Generation (RAG) chatbot that can answer questions based on the complete works of William Shakespeare. It runs entirely locally using open-source components: [Ollama](https://ollama.com/) for language models and embedding generation, [ChromaDB](https://www.trychroma.com/) for vector storage, and [LangChain](https://www.langchain.com/) to orchestrate the retrieval and response generation.

---

## Features

- Ingests a plain text file (`input.txt`) containing Shakespeare's complete works
- Splits and embeds text using `nomic-embed-text` locally via Ollama
- Stores document embeddings in ChromaDB (vector store)
- Uses a local `llama3.2` model to generate answers grounded in retrieved text
- Caches both embeddings and vector DB to avoid redundant computation
- Command-line interface for querying

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/bardbot.git
cd bardbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Ollama
Follow instructions at [ollama.com](https://ollama.com) or run:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 4. Pull the required models
```bash
ollama pull nomic-embed-text:v1.5
ollama pull llama3.2
```

### 5. Add the input text
Place Shakespeare’s complete works in a file named `input.txt` in the root directory.

You can use [Project Gutenberg's Shakespeare collection](https://www.gutenberg.org/ebooks/100) for this.

---

## Usage

Run the chatbot by executing:
```bash
python script.py
```

You will be prompted to enter a query. The chatbot will return context-aware answers using its knowledge of Shakespeare's plays.

Example:
```
You: who are the main characters in the romeo and juliet play

ShakespeareGPT: The main characters in the Romeo and Juliet play include:

1. Romeo Montague - The male protagonist of the story, a member of the Montague family and in love with Juliet.
2. Juliet Capulet - The female protagonist of the story, a member of the Capulet family and in love with Romeo.
3. Tybalt - A character from the Capulet family who is Juliet's cousin and has a long-standing feud with Romeo.
4. Friar Lawrence - A wise and understanding priest who marries Romeo and Juliet in secret.
5. Nurse - Juliet's loyal nurse who serves as a confidante to Juliet and provides information about her life.

Note: There are also other significant characters in the play, such as Lord Capulet, Lady Capulet, Mercutio, and Paris, but these five are generally considered the main characters.

```

---

## Code Overview

The main logic resides in `script.py`. It is organized as an object-oriented pipeline with the following structure:

### `RAGChatbot` class
Encapsulates the entire retrieval-augmented generation workflow.

- **`__init__()`**: Initializes paths and internal components
- **`load_text()`**: Loads and reads the `input.txt` file
- **`split_text()`**: Uses LangChain's `RecursiveCharacterTextSplitter` to chunk the input
- **`generate_embeddings()`**: Uses `nomic-embed-text` model from Ollama to generate and save embeddings
- **`build_vectorstore()`**: Creates or loads a ChromaDB vector store from embeddings
- **`initialize_chain()`**: Sets up the `RetrievalQA` chain using a local `llama3.2` model
- **`chat_loop()`**: Starts the input loop for user queries

### `OllamaEmbedder` class
Extends `langchain.embeddings.base.Embeddings` to wrap embedding calls via Ollama.

---

## Requirements

- Python 3.10 or higher
- Ollama installed and running (`ollama serve`)
- Models:
  - `nomic-embed-text:v1.5`
  - `llama3.2`

---

## File Structure

```
bardbot/
├── script.py              
├── input.txt              
├── embeddings.pkl       
├── chroma_db/            
├── requirements.txt    
└── README.md          
```

---

## Notes

- You can delete `embeddings.pkl` or `chroma_db/` to force a rebuild.
- Make sure the Ollama server is running in the background before starting the script.
- Uses `tqdm` to show embedding progress for large files.

---
