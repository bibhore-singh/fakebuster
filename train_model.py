"""
Fake News Detection - Model Training Pipeline
Model: Passive Aggressive Classifier (PAC)
Vectorization: TF-IDF Vectorizer
Evaluation: Accuracy, Confusion Matrix, Classification Report
Source Credibility: Multi-article Source Visibility Weighting
Author: Bibhore Raj
"""

import os
import re
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def clean_text(text):
    """Normalize and clean input news text."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def train_pipeline(dataset_path='dataset/train.csv', output_dir='.'):
    """Load dataset, train TF-IDF + Passive Aggressive Classifier, evaluate, and save artifacts."""
    print("=" * 60)
    print("🚀 FAKE NEWS DETECTION: TRAINING PASSIVE AGGRESSIVE CLASSIFIER")
    print("=" * 60)
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
        
    df = pd.read_csv(dataset_path)
    print(f"[*] Loaded dataset: {len(df)} samples")
    
    # Combine title, author, and text for rich contextual representation
    df['combined'] = df['title'].fillna('') + ' ' + df['author'].fillna('') + ' ' + df['text'].fillna('')
    df['cleaned'] = df['combined'].apply(clean_text)
    
    X = df['cleaned']
    y = df['label']
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"[*] Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_df=0.75,
        min_df=1,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    
    print("[*] Fitting TF-IDF Vectorizer...")
    tfidf_train = vectorizer.fit_transform(X_train)
    tfidf_test = vectorizer.transform(X_test)
    
    # Passive Aggressive Classifier
    pac = PassiveAggressiveClassifier(
        C=0.5,
        max_iter=50,
        random_state=42,
        loss='hinge',
        n_jobs=-1
    )
    
    print("[*] Training Passive Aggressive Classifier...")
    pac.fit(tfidf_train, y_train)
    
    # Evaluation
    y_pred = pac.predict(tfidf_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    
    print("\n" + "=" * 40)
    print(f"🎯 EVALUATION ACCURACY: {acc * 100:.2f}%")
    print("=" * 40)
    print("\n[+] Confusion Matrix [ [TN, FP], [FN, TP] ]:")
    print(f"    Real (0) predicted Real (0) [TN]: {cm[0][0]}")
    print(f"    Real (0) predicted Fake (1) [FP]: {cm[0][1]}")
    print(f"    Fake (1) predicted Real (0) [FN]: {cm[1][0]}")
    print(f"    Fake (1) predicted Fake (1) [TP]: {cm[1][1]}")
    
    print("\n[+] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Reliable (0)', 'Unreliable (1)']))
    
    # Save model and vectorizer
    model_path = os.path.join(output_dir, 'model.pkl')
    vector_path = os.path.join(output_dir, 'vector.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(pac, f)
    with open(vector_path, 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print(f"[✓] Saved model artifact: {model_path}")
    print(f"[✓] Saved vectorizer artifact: {vector_path}")
    
    return pac, vectorizer

def calculate_source_visibility_weight(history_predictions, decay_factor=0.85):
    """
    Computes a social media visibility weight (0.0 to 1.0) for a news source
    based on historical article predictions.
    
    history_predictions: list of binary flags (0=real, 1=fake) or confidence values.
    Returns: visibility_weight (float between 0.05 and 1.0)
    """
    if not history_predictions:
        return 1.0
    
    # Exponential moving average of fake ratio
    fake_count = sum(1 for p in history_predictions if p == 1)
    total = len(history_predictions)
    fake_ratio = fake_count / total
    
    if fake_ratio >= 0.7:
        # High-confidence fake source: penalize heavily
        visibility = max(0.05, 1.0 - (fake_ratio ** 2 * 1.2))
    elif fake_ratio >= 0.3:
        # Suspicious source: moderate throttle
        visibility = max(0.2, 1.0 - (fake_ratio * 0.9))
    else:
        # Reputable source: near full visibility
        visibility = 1.0 - (fake_ratio * 0.2)
        
    return round(min(1.0, max(0.05, visibility)), 3)

if __name__ == '__main__':
    train_pipeline()
