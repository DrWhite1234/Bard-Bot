import os
import pickle
from tqdm import tqdm

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

import ollama


class OllamaEmbedder(Embeddings):
    def embed_documents(self, texts):
        return [ollama.embeddings(model='nomic-embed-text:v1.5', prompt=t)['embedding'] for t in tqdm(texts, desc="Embedding Chunks")]

    def embed_query(self, text):
        res = ollama.embeddings(model='nomic-embed-text:v1.5', prompt=text)
        embedding = res.get("embedding", [])
        if not embedding:
            raise ValueError(f"Failed to get embedding for query: {text}")
        return embedding 

class ShakespeareRAG:
    def __init__(self, input_path="input.txt", persist_dir="./chroma_db", embed_cache="embeddings.pkl"):
        self.input_path = input_path
        self.persist_dir = persist_dir
        self.collection_name = "shakespeare"
        self.embed_cache = embed_cache
        self.embedder = OllamaEmbedder()
        self.embedding_func = self.embedder
        self.texts = []
        self.embeddings = []
        self.vectorstore = None
        self.qa_chain = None

    def load_text(self):
        print("Opening text input...")
        with open(self.input_path, "r", encoding="utf-8") as f:
            return f.read()

    def split_text(self, raw_text):
        print("Splitting text into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        documents = splitter.create_documents([raw_text])
        self.texts = [doc.page_content for doc in documents]

    def generate_embeddings(self):
        if os.path.exists(self.embed_cache):
            print("Loading embeddings...")
            with open(self.embed_cache, "rb") as f:
                self.embeddings = pickle.load(f)
        else:
            print("Generating embeddings...")
            self.embeddings = self.embedder.embed_documents(self.texts)
            with open(self.embed_cache, "wb") as f:
                pickle.dump(self.embeddings, f)
            print("Embeddings saved.")

    def build_vectorstore(self):
        print("Checking for existing vector store...")
        if os.path.exists(self.persist_dir) and os.path.exists(os.path.join(self.persist_dir, "chroma.sqlite3")):
            print("Loading existing vector store...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embedding_func,
                collection_name=self.collection_name
            )
        else:
            print("Creating new vector store...")
            self.vectorstore = Chroma.from_texts(
                texts=self.texts,
                embedding=self.embedding_func,
                persist_directory=self.persist_dir,
                collection_name=self.collection_name
            )
            print("Vector store created.")


    def setup_qa_chain(self):
        print("Initializing chain...")
        retriever = self.vectorstore.as_retriever(search_type="similarity", k=5)
        llm = Ollama(model="llama3.2")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

    def chat_loop(self):
        print("\nAsk anything about Shakespeare:")
        while True:
            query = input("\nYou: ")
            result = self.qa_chain.invoke({"query": query})
            print(f"\nShakespeareGPT: {result['result']}")


def main():
    rag = ShakespeareRAG()

    raw_text = rag.load_text()
    rag.split_text(raw_text)
    rag.generate_embeddings()
    rag.build_vectorstore()
    rag.setup_qa_chain()
    rag.chat_loop()


if __name__ == "__main__":
    main()
