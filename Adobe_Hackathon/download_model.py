from sentence_transformers import SentenceTransformer

# This is a good, small, and effective model for this task.
model_name = 'all-MiniLM-L6-v2' 

print(f"Downloading model: {model_name}...")

# This line downloads the model from the internet and saves it.
model = SentenceTransformer(model_name)

# This line saves all the necessary files into a new folder named 'local-model'.
model.save('./local-model')

print("Model downloaded and saved successfully into the './local-model' folder!")