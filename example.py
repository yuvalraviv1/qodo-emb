from sentence_transformers import SentenceTransformer

# Define the list of model names
model_names = [
    "Qodo/Qodo-Embed-1-1.5B",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2"
]

# The sentences to encode
sentences = [
    'accumulator = sum(item.value for item in collection)',  
    'result = reduce(lambda acc, curr: acc + curr.amount, data, 0)',  
    'matrix = [[i*j for j in range(n)] for i in range(n)]'  ,
    'a = sum(items)',
    'def add(list): s=0; for i in list: s+=i;  return s',
    # New snippets:
    'squares_of_evens = [x*x for x in range(10) if x % 2 == 0]',
    'class Greeter:\\n def __init__(self, name):\\n  self.name = name\\n def greet(self):\\n  return f"Hello, {self.name}!"',
    'def factorial(n):\\n if n == 0:\\n  return 1\\n else:\\n  return n * factorial(n-1)',
    'reversed_string = "hello"[::-1]',
    'my_dict = {"a": 1, "b": 2}; my_dict["c"] = 3',
    '# This is a comment\\nx = 10 + 5',
    'def find_max(numbers):\\n max_val = numbers[0]\\n for x in numbers:\\n  if x > max_val:\\n   max_val = x\\n return max_val'
]

# Loop through each model
for model_name in model_names:
    print(f"\n---- Running Model: {model_name} ----")

    # Instantiate the SentenceTransformer with the current model name
    # Download from the 🤗 Hub
    model = SentenceTransformer(model_name)
    
    # Run inference
    # Encode the sentences
    embeddings = model.encode(sentences)
    print("Embeddings:")
    print(embeddings)
    # [3, 1536] # This comment might need adjustment or removal depending on actual output dimensions

    # Get the similarity scores for the embeddings
    similarities = model.similarity(embeddings, embeddings)
    print("Similarity Matrix:")
    print(similarities)
    # [3, 3] # This comment might need adjustment or removal
