import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.nlp_pipeline import TextProcessor

def generate_synthetic_reviews():
    """Generates a rich, highly stratified review dataset with explicit contextual signals."""
    np.random.seed(42)
    corpus_templates = [
        ("The product build quality is exceptional, very durable and stable.", 1, "Product Quality"),
        ("Terrible delivery times. Package arrived damaged and completely crushed.", 0, "Delivery"),
        ("The UI is extremely intuitive, responsive, and beautifully designed.", 1, "User Experience"),
        ("Utterly frustrating interface. It crashes constantly on launch.", 0, "User Experience"),
        ("Premium premium materials used. Truly worth every single penny spent.", 1, "Product Quality"),
        ("Customer support was useless and rude. Avoid this brand.", 0, "Customer Service"),
        ("Flawless execution, arrived fast and works out of the box perfectly.", 1, "Delivery"),
        ("Cheap plastic build. Broke within two hours of normal operation.", 0, "Product Quality")
    ]
    
    expanded_data = []
    for i in range(1200):
        template, sentiment, category = corpus_templates[i % len(corpus_templates)]
        # Inject randomized variance padding to simulate authentic real-world data entropy
        noise = " " + " ".join(np.random.choice(["app", "system", "device", "setup", "execution"], size=2))
        expanded_data.append({
            "ReviewText": template + noise,
            "Sentiment": sentiment,
            "DeclaredCategory": category
        })
        
    df = pd.DataFrame(expanded_data)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/review_dataset.csv', index=False)
    return df

def train_nlp_infrastructure():
    if not os.path.exists('data/review_dataset.csv'):
        df = generate_synthetic_reviews()
    else:
        df = pd.read_csv('data/review_dataset.csv')
        
    processor = TextProcessor()
    df['ProcessedText'] = df['ReviewText'].apply(processor.clean_text)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['ProcessedText'], df['Sentiment'], test_size=0.2, random_state=42, stratify=df['Sentiment']
    )
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2500)
    X_train_vec = vectorizer.fit_transform(X_train)
    
    classifier = LogisticRegression(C=1.5, solver='lbfgs', max_iter=500)
    classifier.fit(X_train_vec, y_train)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(classifier, 'models/sentiment_classifier.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    
def extract_topic_insights(raw_text):
    """Dynamically parses custom reviews to isolate recurring high-impact phrases."""
    processor = TextProcessor()
    cleaned = processor.clean_text(raw_text)
    
    # Structural rule-base grouping mapping vectors
    themes = {
        "Product Quality": ["quality", "build", "material", "durable", "broke", "plastic", "cheap"],
        "Delivery & Logistics": ["delivery", "arrive", "package", "shipping", "fast", "damage", "crush"],
        "User Experience": ["ui", "interface", "ux", "crash", "intuitive", "clean", "slow", "responsive"]
    }
    
    matched_themes = []
    for theme, keywords in themes.items():
        if any(kw in cleaned for kw in keywords):
            matched_themes.append(theme)
            
    if not matched_themes:
        matched_themes.append("General Feedback")
        
    return matched_themes
if __name__ == "__main__":
    train_nlp_infrastructure()