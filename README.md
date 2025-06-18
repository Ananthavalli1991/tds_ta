# tds_ta
This repository is created for answering students questions based course content and discourse posts
# 🤖 TDS Virtual TA API

A FastAPI-based Virtual Teaching Assistant for the IITM BSc TDS course.  
It can answer student questions based on text and image input using OpenAI embeddings and LanceDB for similarity search.

---

## 🔍 Features

- 📄 Semantic search and embed images using OpenAI's `text-embedding-3-small`
- 🖼️ Image support via `gpt-4o-mini` to describe 
- 🔎 Fast vector similarity search with LanceDB
- 🔗 Relevant link extraction from stored metadata
- 🚀 Easily deployable on Render.com or any Python server

---

## 🗂️ Project Structure

tds_virtual_ta/
├── main.py # FastAPI application
├── store_embeddings.py # Loads .npz data into LanceDB
├── embed_chunks.npz # Text content + embeddings + metadata
├── img_emb.npz # Image descriptions + embeddings + urls
├── db/ # LanceDB storage (auto-generated)
├── requirements.txt # Project dependencies
├── render.yaml # Render deployment config
├── .gitignore # Ignores db/, env, etc.
└── README.md # You're reading this :)

---

## ⚙️ Setup (Local)

### 1. Install dependencies
```bash
pip install -r requirements.txt

###**2. Load .npz files into LanceDB**
python store_embeddings.py

###**3. Run the API locally**
uvicorn main:app --reload

###**4.Example request (JSON):**
{
  "question": "Should I use gpt-4o-mini which AI proxy supports, or gpt3.5 turbo?",
  "image": "BASE64_ENCODED_IMAGE_STRING"
}

###**Example response:**
{
  "answer": "You must use gpt-3.5-turbo...",
  "links": [
    {
      "url": "https://discourse.example.com/thread/123",
      "text": "Clarification from instructor"
    }
  ]
}

