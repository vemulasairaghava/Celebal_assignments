"""Inspect all chunks stored in the FAISS vector database."""
import pickle
import sys
import io

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open("vectorstore/index.pkl", "rb") as f:
    data = pickle.load(f)

# Find the docstore and id mapping
docstore = None
id_map = None
for item in data:
    cls_name = type(item).__name__
    if cls_name == "InMemoryDocstore":
        docstore = item
    elif isinstance(item, dict):
        id_map = item

print(f"Total chunks stored: {len(id_map)}")
print("=" * 80)

for i, (idx, doc_id) in enumerate(sorted(id_map.items())):
    doc = docstore.search(doc_id)
    meta = doc.metadata
    # Replace non-ASCII for safe terminal display
    content_preview = doc.page_content[:200].encode('ascii', errors='replace').decode('ascii')
    print(f"\n--- Chunk {i+1} of {len(id_map)} ---")
    print(f"  Chunk ID:    {meta.get('chunk_id', 'N/A')}")
    print(f"  File:        {meta.get('filename', '?')}")
    print(f"  Page:        {meta.get('page_number', '?')}")
    print(f"  Chunk Index: {meta.get('chunk_index', '?')}")
    print(f"  Length:      {len(doc.page_content)} chars")
    print(f"  Content:     {content_preview}...")
    print()
