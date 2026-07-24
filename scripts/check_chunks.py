import chromadb

client = chromadb.PersistentClient(path="data/knowledge_base/chroma_store")
collection = client.get_collection("aml_typologies")

for source in ["FIN-2011-A016.pdf", "FIN-2016-A003.pdf"]:
    results = collection.get(where={"source": source})
    print(f"{source}: {len(results['ids'])} chunks")