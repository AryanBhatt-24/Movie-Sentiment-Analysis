🎬 Movie Sentiment Analysis

A Flask web application that predicts whether a movie review is positive or negative using Natural Language Processing (NLP) and Machine Learning. This project demonstrates deploying a trained ML model with a premium web interface for real-time predictions.

🔹 Features

Predict positive or negative sentiment for any movie review.

Clean, modern, and professional frontend.

Uses TF-IDF vectorization and Naive Bayes / Logistic Regression for accurate predictions.

Displays recent predictions for better tracking.

Fully responsive design for desktop and mobile.

🔹 Tech Stack

Frontend: HTML, CSS, Bootstrap

Backend: Python, Flask

Machine Learning: Scikit-learn, Naive Bayes, Logistic Regression

NLP: NLTK for text preprocessing and tokenization

🔹 How It Works

The user enters a movie review in the input box.

The text is preprocessed (lowercased, cleaned, lemmatized).

The TF-IDF vectorizer converts the text into numerical features.

The trained ML model predicts if the sentiment is positive or negative.

The result is displayed on the web page in a clean, colored badge.

🔹 Example

Input review:

"I absolutely loved this movie! The acting and storyline were fantastic."


Prediction:

Positive Review

🔹 Project Structure
Movie-Sentiment-Analysis/
│
```├─ app.py                  # Flask backend```
```├─ sentiment_model.pkl      # Trained ML model```
```├─ tfidf_vectorizer.pkl     # TF-IDF vectorizer```
├─ templates/
│   └─ index.html           # Frontend HTML
├─ static/                  # Optional CSS/images for styling
└─ README.md

🔹 Future Enhancements

Deploy the app on Heroku, Streamlit, or AWS for live access.

Add real-time sentiment analysis as the user types.

Implement dark mode for a modern look.

Upgrade to deep learning models (LSTM or BERT) for better accuracy.

🔹 Author

Aryan Bhatt

GitHub: AryanBhatt-24

Email: aryanbhatt502@gmail.com
