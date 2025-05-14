from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("Qodo/Qodo-Embed-1-1.5B")
# Run inference
sentences = [
    'accumulator = sum(item.value for item in collection)',  
    'result = reduce(lambda acc, curr: acc + curr.amount, data, 0)',  
    'matrix = [[i*j for j in range(n)] for i in range(n)]'  ,
    'a = sum(items)',
    'def add(list): s=0; for i in list: s+=i;  return s',
]
embeddings = model.encode(sentences)
print(embeddings)
# [3, 1536]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)

# [3, 3]
