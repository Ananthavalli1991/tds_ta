import numpy as np
import lancedb

# Load text data
text_data = np.load("c:/Users/DELL/embed_chunks.npz", allow_pickle=True)
texts = text_data["texts"]
text_embeddings = text_data["embeddings"]
text_metadata = text_data["metadatas"]

# Load image data
img_data = np.load("c:/Users/DELL/image_emb.npz", allow_pickle=True)
img_urls = img_data["urls"]
img_embeddings = img_data["embeddings"]
img_chunk_idx = img_data["chunk_indices"]

# Format text records
text_records = [{
    "type": "text",
    "content": texts[i],
    "embedding": text_embeddings[i].tolist(),
    "meta": text_metadata[i]
} for i in range(len(texts))]

# Format image records
image_records = [{
    "type": "image",
    "content": img_urls[i],
    "embedding": img_embeddings[i].tolist(),
    "chunk_id": int(img_chunk_idx[i]),
    "meta": {"url": img_urls[i]}
} for i in range(len(img_urls))]

# Combine records
all_records = text_records + image_records

# Write to LanceDB
db = lancedb.connect("db")
if "tds" in db.table_names():
    db.drop_table("tds")
table = db.create_table("tds", data=all_records)
print(f"✅ Stored {len(all_records)} records in LanceDB.")
