import numpy as np
from sentence_transformers import SentenceTransformer

class Database:
    def __init__(self):
        # Initialize the sentence transformer model for text embeddings
        self.text_model = SentenceTransformer("all-MiniLM-L6-v2")
        # Simulated database of dress descriptions
        self.dress_descriptions = [
            "Red formal dress with embroidery",
            "Blue casual kurta for women",
            "Black party wear gown",
            "White wedding dress with lace",
            "Green traditional shalwar kameez"
        ]
        # Precompute text embeddings
        self.text_embeddings = self.text_model.encode(self.dress_descriptions, normalize_embeddings=True)
    
    def get_text_features(self):
        # Return the descriptions and their embeddings
        return self.dress_descriptions, self.text_embeddings
