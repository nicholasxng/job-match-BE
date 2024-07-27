from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Download NLTK data files
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

@app.route('/match', methods=['POST'])
def match():
    data = request.json
    resume = data['resume']
    job_description = data['jobDescription']
    
    # Preprocess the text
    resume = preprocess(resume)
    job_description = preprocess(job_description)
    
    # Calculate TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume, job_description])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    # Prepare the response
    response = {
        'similarity': similarity[0][0],
        'match_score': similarity[0][0] * 100  # Convert to percentage
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(port=3000)
