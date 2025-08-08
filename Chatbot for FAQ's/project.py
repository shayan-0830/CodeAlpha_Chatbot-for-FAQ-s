# A simple FAQ chatbot using NLTK for text preprocessing
# and scikit-learn for TF-IDF and cosine similarity.

import nltk
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- 1. DOWNLOAD NLTK RESOURCES ---
# These resources are needed for tokenization and stopword removal.
# They are downloaded once and then can be used offline.
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError as e:
    print("Downloading NLTK resources. This will happen only once.")
    try:
        nltk.download('stopwords')
        nltk.download('punkt')
    except Exception as download_error:
        print("Error downloading NLTK resources:", download_error)
        print("Original exception:", e)
        raise

# --- 2. DEFINE FAQ DATA ---
# This is a dictionary of questions and their corresponding answers.
# The keys are the questions the chatbot will try to match.
faq_data = {
    "What is the return policy?": "Our return policy allows for returns within 30 days of purchase, provided the item is in its original condition with a receipt.",
    "How can I track my order?": "You can track your order by logging into your account and visiting the 'Order History' page. A tracking number will be provided there.",
    "Do you offer international shipping?": "Yes, we ship to over 50 countries worldwide. Shipping costs and times vary by destination.",
    "What payment methods are accepted?": "We accept all major credit cards, PayPal, and Apple Pay.",
    "How do I contact customer support?": "You can contact our customer support team via email at support@example.com or by calling our toll-free number.",
    "Where are you located?": "Our main office is in New York City, but we operate globally.",
    "How do I reset my password?": "You can reset your password by clicking 'Forgot Password' on the login page and following the instructions sent to your email.",
    "What are your business hours?": "Our business hours are Monday to Friday, 9:00 AM to 5:00 PM EST.",
    "Can I change my shipping address?": "You can change your shipping address before the order is processed. Please contact us immediately if you need to make a change.",
}

# Get the list of all FAQ questions
faq_questions = list(faq_data.keys())

# --- 3. TEXT PREPROCESSING FUNCTION ---
# This function cleans and tokenizes the input text.
def preprocess_text(text):
    """
    Cleans text by converting to lowercase, removing punctuation, and tokenizing.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stopwords (common words like 'a', 'the', 'is')
    stopwords = nltk.corpus.stopwords.words('english')
    filtered_tokens = [word for word in tokens if word not in stopwords]
    return ' '.join(filtered_tokens)

# --- 4. TF-IDF VECTORIZATION ---
# This converts the text data into a numerical format (vectors)
# that can be used for similarity calculations.
# We create and train the vectorizer on the FAQ questions.
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform([preprocess_text(q) for q in faq_questions])

# --- 5. CHATBOT LOGIC: FINDING THE BEST MATCH ---
def get_best_match(user_question):
    """
    Takes a user question, finds the most similar FAQ question, and returns the answer.
    """
    # Preprocess the user's question
    processed_user_question = preprocess_text(user_question)

    # If the user's question is empty after preprocessing, return a default response
    if not processed_user_question:
        return "I'm sorry, I didn't catch that. Can you please rephrase your question?"

    # Vectorize the user's question using the same vectorizer instance
    user_vector = vectorizer.transform([processed_user_question])

    # Calculate cosine similarity between user question and all FAQ questions
    # `faq_vectors` is a sparse matrix, so we can't directly compare it with the user_vector.
    # We can use the cosine_similarity function from scikit-learn.
    similarity_scores = cosine_similarity(user_vector, faq_vectors)

    # Get the index of the best match (the highest score)
    best_match_index = np.argmax(similarity_scores)
    best_score = similarity_scores[0, best_match_index]

    # Define a similarity threshold to avoid giving a wrong answer for a bad match
    SIMILARITY_THRESHOLD = 0.2

    if best_score > SIMILARITY_THRESHOLD:
        # Get the original question and its answer
        matched_question = faq_questions[best_match_index]
        matched_answer = faq_data[matched_question]
        print(f"I found a similar question: '{matched_question}'")
        return matched_answer
    else:
        return "I'm sorry, I don't have an answer for that. Please try asking a different question."

# --- 6. SIMPLE CHAT UI LOOP ---
print("FAQ Chatbot: Hello! I can answer questions about our product.")
print("Type 'quit' or 'exit' to end the chat.")
print("-" * 50)

while True:
    print("Waiting for your question...")
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("FAQ Chatbot: Goodbye!")
        break
    
    response = get_best_match(user_input)
    print("FAQ Chatbot:", response)
    print("-" * 50)
