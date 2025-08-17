import os
import gradio as gr
import pandas as pd
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter


load_dotenv()

BOOKS_CSV = "books_cleaned.csv"
CHROMA_DIR = "chroma_db"



books = pd.read_csv(BOOKS_CSV)



embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


should_build = not os.path.exists(CHROMA_DIR)
if not should_build:
    db_temp = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_function)
    should_build = db_temp._collection.count() == 0

if should_build:
    print("🧠 Building Chroma DB from scratch...")

    documents = []
    for i, row in books.iterrows():
        title = str(row.get("title", "Unknown Title"))
        author = str(row.get("authors", "Unknown Author"))
        content = str(row.get("description")).strip() if pd.notna(row.get("description")) else ""

        if not content or content.lower() == "nan":
            continue

        doc = Document(
            page_content=content,
            metadata={"title": title, "author": author}
        )
        documents.append(doc)

    print(f"✅ Valid documents to index: {len(documents)}")

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(documents)
    print(f"🔎 Total chunks created: {len(split_docs)}")

    db = Chroma.from_documents(split_docs, embedding_function, persist_directory=CHROMA_DIR)

    print("✅ Chroma DB saved to disk.")
else:
    print("📁 Loading existing Chroma DB...")
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_function)
    print(f"📚 DB loaded. Total documents: {db._collection.count()}")


def search_books(query, k=3):
    print("🔍 New query:", query)
    results = db.similarity_search(query, k=k)
    print(f"📦 Found {len(results)} results")

    if not results:
        return "❌ No results found. Try a different query."

    formatted = [
        f"📖 **{r.metadata.get('title', 'Unknown')}** by {r.metadata.get('author', 'Unknown')}\n\n{r.page_content}"
        for r in results
    ]
    return "\n\n---\n\n".join(formatted)


iface = gr.Interface(
    fn=search_books,
    inputs=[
        gr.Textbox(label="Search Query", placeholder="e.g. A book about war and love", lines=2),
        gr.Slider(1, 10, value=3, step=1, label="Number of Results")
    ],
    outputs=gr.Textbox(label="Search Results", lines=20),
    title="📚 Book Recommender",
    description="Search through a book collection using AI and vector similarity."
)

iface.launch()
