import os
from pathlib import Path

# Step 1: PDF Loading
from langchain_community.document_loaders import PyPDFLoader

# Step 2: Chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 3: TF-IDF Vector Store (cosine similarity)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 4: Gemini LLM (new google-genai SDK)
from google import genai
from dotenv import load_dotenv


# Load API key from RAG/.env
_here     = Path(__file__).parent          # -> .../RAG/
_env_path = _here / ".env"
load_dotenv(dotenv_path=_env_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found.\n"
        "Get a free key at https://aistudio.google.com/app/apikey\n"
        "Then add it to RAG/.env:\n"
        "  GEMINI_API_KEY=your_key_here"
    )

# Configure Gemini client
_client = genai.Client(api_key=GEMINI_API_KEY)

print("[RAG] Gemini API configured [OK]")


# Step 1: Load PDF
PDF_PATH = _here / "Project.pdf"

print("[RAG] Loading PDF:", PDF_PATH)
loader = PyPDFLoader(str(PDF_PATH))
pages  = loader.load()


# Step 2: Chunk the pages
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
docs     = splitter.split_documents(pages)
chunks   = [d.page_content for d in docs]

print(f"[RAG] {len(pages)} pages -> {len(chunks)} chunks")


# Step 3: Build TF-IDF index (Vector Database equivalent)
_vectorizer   = TfidfVectorizer(stop_words="english")
_tfidf_matrix = _vectorizer.fit_transform(chunks)

print("[RAG] TF-IDF vector index ready [OK]")


# Custom exception for rate limiting
class RateLimitError(Exception):
    pass


# Steps 4 & 5: Retrieve + Generate Answer
def answer_question(question: str, top_k: int = 5) -> str:
    """
    RAG pipeline:
      1. Convert question to TF-IDF vector
      2. Cosine similarity search -> retrieve top_k relevant chunks
      3. Pass retrieved context to Gemini LLM
      4. Return grounded answer (only from PDF content)
    """

    # Retrieval (Similarity Search)
    q_vec   = _vectorizer.transform([question])
    scores  = cosine_similarity(q_vec, _tfidf_matrix).flatten()
    top_idx = scores.argsort()[::-1][:top_k]

    # Return top chunks (remove strict score > 0 filter so broad queries still retrieve context)
    context = "\n\n".join(chunks[i] for i in top_idx)

    if not context.strip():
        return "I couldn't find relevant information in the project documentation for that question."

    # Generation (LLM Answer from Retrieved Context)
    prompt = (
        "You are a helpful assistant for the GreenPredict-AI project.\n"
        "Answer the user's question using ONLY the context below.\n"
        "Do NOT use any knowledge outside of this context.\n"
        "Keep your answer short: 2 to 4 sentences maximum.\n"
        "If the context does not contain the answer, say: "
        "'I don't have that information in the project documentation.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    try:
        response = _client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return (response.text or "I couldn't generate an answer. Please try again.").strip()

    except Exception as exc:
        err_str = str(exc)
        # Handle Gemini rate limiting (429 / RESOURCE_EXHAUSTED)
        if "429" in err_str or "resource_exhausted" in err_str.lower():
            raise RateLimitError(
                "The AI is busy right now. Please wait a moment and try again."
            )
        raise
