from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

COLLECTION_NAME = "vinni_profile"

# Embeddings model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Production-grade chunker
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=[
        "\n================",  # your section dividers
        "\n\n",
        "\n",
        " ",
        ""
    ]
)


# ── CHUNKING ────────────────────────────────────────────────────
def load_and_chunk_profile() -> list[str]:
    with open("data/vidhya_profile.txt", "r") as f:
        content = f.read()

    chunks = splitter.split_text(content)

    # Clean separator noise
    cleaned = []
    for chunk in chunks:
        lines = [l for l in chunk.split("\n") if not set(l.strip()).issubset({"=", "-", " ", ""})]
        cleaned_chunk = "\n".join(lines).strip()
        if len(cleaned_chunk) > 80:
            cleaned.append(cleaned_chunk)

    return cleaned


# ── EVALUATION ──────────────────────────────────────────────────
def evaluate_chunks(chunks: list[str]):
    lengths = [len(c) for c in chunks]
    avg = sum(lengths) / len(lengths)
    too_small = [i+1 for i, c in enumerate(chunks) if len(c) < 80]
    mid_sentence = [
        i+1 for i, c in enumerate(chunks)
        if c[0].islower() and not c.startswith("iOS")
    ]

    print(f"\n📊 Chunk Evaluation Report")
    print(f"Total chunks     : {len(chunks)}")
    print(f"Avg chunk size   : {avg:.0f} chars")
    print(f"Smallest chunk   : {min(lengths)} chars")
    print(f"Largest chunk    : {max(lengths)} chars")
    print(f"Too small (<80)  : chunks {too_small if too_small else 'none'}")
    print(f"Mid-sentence cuts: chunks {mid_sentence if mid_sentence else 'none'}")

    print(f"\n✅ Good to embed!" if not too_small and not mid_sentence else "\n⚠️  Fix issues before embedding")


# ── INDEXING (run once) ─────────────────────────────────────────
# What LangChain replaces (manual version):
#
# 1. qdrant.create_collection(collection_name, vectors_config=VectorParams(size=1536, distance=Distance.COSINE))
# 2. for chunk in chunks:
#        vector = openai_client.embeddings.create(model="text-embedding-3-small", input=chunk)
#        points.append(PointStruct(id=uuid4(), vector=vector, payload={"text": chunk}))
# 3. qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
#
# LangChain does all three steps in one call:
def index_profile():
    chunks = load_and_chunk_profile()
    evaluate_chunks(chunks)

    vectorstore = QdrantVectorStore.from_texts(
        texts=chunks,
        embedding=embeddings,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=COLLECTION_NAME,
        force_recreate=True  # wipes and re-indexes fresh every time
    )

    print(f"\n✅ Indexed {len(chunks)} chunks into Qdrant via LangChain!")
    return vectorstore


# ── RETRIEVAL ───────────────────────────────────────────────────
# What LangChain replaces (manual version):
#
# question_vector = openai_client.embeddings.create(model="text-embedding-3-small", input=question)
# results = qdrant.search(collection_name=COLLECTION_NAME, query_vector=question_vector, limit=top_k)
# return [r.payload["text"] for r in results]
#
# LangChain does it in one call:
def get_vectorstore() -> QdrantVectorStore:
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name=COLLECTION_NAME
    )


def retrieve(question: str, top_k: int = 3) -> list[str]:
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search(question, k=top_k)
    return [r.page_content for r in results]


# ── TEST ────────────────────────────────────────────────────────
if __name__ == "__main__":
    chunks = load_and_chunk_profile()
    index_profile()

    test_questions = [
        "What iOS projects has Vidhya built?",
        "What are Vidhya's backend skills?",
        "Tell me about Vidhya's research and publications",
        "What is Vidhya's experience at Target Corporation?",
        "What AI and ML technologies does Vidhya know?",
    ]

    for question in test_questions:
        print(f"\n🔍 Query: {question}")
        results = retrieve(question)
        print(f"Top result: {results[0][:150]}...")