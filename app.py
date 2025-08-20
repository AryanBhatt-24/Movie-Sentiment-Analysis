from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

app = Flask(__name__)

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def preprocess(text):
    text = text.lower()
    text = re.sub('[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    cleaned = preprocess(review)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)[0]

    result = "Positive Review 😊" if pred == "positive" else "Negative Review 😡"
    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
