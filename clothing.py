import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset of clothing item descriptions
dataset = [
    "This is a blue T-shirt made of cotton.",
    "A red dress with floral print.",
    "Black jeans made of stretchable fabric.",
    "A white blouse with lace trim.",
    # Add more clothing item descriptions to the dataset
]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer on the dataset
X = vectorizer.fit_transform(dataset)

def get_similar_items(input_text, n=5):
    # Preprocess the input text
    input_vector = vectorizer.transform([input_text])

    # Compute cosine similarities between the input and dataset
    similarities = cosine_similarity(input_vector, X)

    # Get the indices of top-N most similar items
    top_indices = similarities.argsort()[0][::-1][:n]

    # Get the URLs of the top-N most similar items (dummy URLs for illustration)
    similar_items = [
        {"url": "https://example.com/item1"},
        {"url": "https://example.com/item2"},
        # Add more URLs and relevant information here based on the dataset
    ]

    return similar_items

# Example usage
input_text = "I'm looking for a blue T-shirt."
result = get_similar_items(input_text)

# Convert the result to JSON
json_result = json.dumps(result)

# Print the JSON result
print(json_result)
