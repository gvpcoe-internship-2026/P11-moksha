from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# Load a pretrained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Two sentences
sentence1 = "software engineer with Python experience"
sentence2 = "looking for Python developer"

# Generate embeddings
embedding1 = model.encode(sentence1, convert_to_tensor=True)
embedding2 = model.encode(sentence2, convert_to_tensor=True)

# Calculate cosine similarity
similarity = cos_sim(embedding1, embedding2)

print("Sentence 1:", sentence1)
print("Sentence 2:", sentence2)
print(f"Cosine Similarity: {similarity.item():.4f}")