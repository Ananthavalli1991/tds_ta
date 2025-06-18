from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import numpy as np
import base64
import os
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Load env
load_dotenv()
client = OpenAI(api_key=os.getenv("AIPipe_TOKEN") or os.getenv("OPENAI_API_KEY"))

# Load npz files
text_data = np.load("embed_chunks.npz", allow_pickle=True)
texts = text_data["texts"]
text_embeddings = text_data["embeddings"]
text_metadata = text_data["metadatas"]

img_data = np.load("image_emb.npz", allow_pickle=True)
img_urls = img_data["urls"]
img_embeddings = img_data["embeddings"]
img_chunk_idx = img_data["chunk_indices"]

# Combine all embeddings + metadata
all_embeddings = np.vstack((text_embeddings, img_embeddings))
all_contents = list(texts) + list(img_urls)
all_metadata = list(text_metadata) + [{"url": u} for u in img_urls]

# FastAPI
app = FastAPI()

class QARequest(BaseModel):
    question: str
    image: str = None

def get_text_embedding(text):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return res.data[0].embedding

def describe_image(base64_image):
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4.1-nano",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url", "image_url": { "url": f"data:image/webp;base64,{base64_image}" }}
                ]
            }],
            max_tokens=300
        )
        return response.choices[0].message.content
    except:
        return ""

def chunk_text(text, chunk_size=500, overlap=50):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def embed_text_chunks(chunks):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks
    )
    vecs = [r.embedding for r in res.data]
    return np.mean(vecs, axis=0)

def search_similar(query_vector, k=5):
    sims = cosine_similarity([query_vector], all_embeddings)[0]
    top_k_idx = sims.argsort()[-k:][::-1]
    return [{"content": all_contents[i], "meta": all_metadata[i], "score": sims[i]} for i in top_k_idx]

@app.post("/api/")
async def qa_endpoint(req: QARequest):
    try:
        text_emb = get_text_embedding(req.question)
        combined_emb = np.array(text_emb)

        if req.image:
            desc = describe_image(req.image)
            if desc:
                chunks = chunk_text(desc)
                image_emb = embed_text_chunks(chunks)
                combined_emb = (combined_emb + image_emb) / 2.0

        results = search_similar(combined_emb)

        context = "\n\n".join([r["content"] for r in results])
        prompt = f"""You are a helpful assistant for the TDS course. Use the context below to answer the question.

        Question: {req.question}

        Context:
        {context}

        Answer:"""

        response = client.chat.completions.create(
            model="openai/gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        answer = response.choices[0].message.content.strip()
        links = []
        for r in results:
            if "url" in r["meta"]:
                links.append({
                    "url": r["meta"]["url"],
                    "text": r["content"][:80]
                })

        return JSONResponse(content={"answer": answer, "links": links})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
