import pandas as pd
import numpy as np
import re
from pathlib import Path
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import cross_validate, KFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. CONFIGURAZIONE FASCE E PERCORSI ---
#BINS = [-np.inf, 50, 1000, 5000, 25000, np.inf]
#LABELS = ['Fallito', 'Debole', 'Buono', 'Forte', 'Virale']
BINS = [-np.inf, 50, 5000, np.inf]
LABELS = ['Fallito', 'Buono', 'Virale']

BASE_PROJECT_PATH = Path.home() / "Desktop" / "scrapingForFood"
FILE_NAME = BASE_PROJECT_PATH / "reddit_food_full_dataset.json"

# Caricamento unico
df = pd.read_json(FILE_NAME)

# --- 2. PULIZIA E CATEGORIZZAZIONE ---
df = df.dropna(subset=['score']) 

# Usiamo pd.cut direttamente invece di una funzione esterna
df['target_class'] = pd.cut(df['score'], bins=BINS, labels=LABELS, include_lowest=True)

# Rimuoviamo eventuali righe che non sono rientrate nei bin (NaN)
df = df.dropna(subset=['target_class'])

# Definizione Feature e Target
X = df[['hour', 'img_blurriness', 'img_warmth', 'img_saturation', 
        'img_centering', 'is_weekend', 'day_of_week', 'title', 'post_text']]
y = df['target_class']

# --- 3. CONFIGURAZIONE NLP ---
stemmer = SnowballStemmer("english")
STOP_WORDS = set(ENGLISH_STOP_WORDS)

def advanced_tokenizer(text):
    if not text: return []
    text = re.sub(r"\[.*?\]", "", text) 
    clean_text = re.sub(r"[^a-z\s]", "", text.lower())
    words = clean_text.split()
    return [stemmer.stem(w) for w in words if w not in STOP_WORDS and len(w) > 2]

# --- 4. PIPELINE DI PREPROCESSING ---
preprocessor = ColumnTransformer(
    transformers=[
        ('title_vec', TfidfVectorizer(tokenizer=advanced_tokenizer, max_features=150, token_pattern=None), 'title'),
        ('text_vec', TfidfVectorizer(tokenizer=advanced_tokenizer, max_features=150, token_pattern=None), 'post_text'),
        ('cat_encode', OneHotEncoder(handle_unknown='ignore'), ['day_of_week']),
        ('num_features', 'passthrough', ['hour', 'img_blurriness', 'img_warmth', 'img_saturation', 'img_centering', 'is_weekend'])
    ]
)

# --- 5. DEFINIZIONE MODELLO ---
"""
def get_classifier_model():
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])
"""
def get_classifier_model():
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=200, 
            random_state=42, 
            n_jobs=-1,
            class_weight='balanced' # <--- AGGIUNGI QUESTO
        ))
    ])
# --- 6. CROSS-VALIDATION ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
print(f"[INFO] Valutazione Classificazione in {len(LABELS)} fasce...")
cv_results = cross_validate(get_classifier_model(), X, y, cv=kf, scoring=['accuracy'])

# --- 7. FIT FINALE E ANALISI ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
final_model = get_classifier_model().fit(X_train, y_train)
y_pred = final_model.predict(X_test)

# --- 8. OUTPUT RISULTATI ---
print("\n" + "="*50)
print("       RISULTATI CLASSIFICAZIONE")
print("="*50)
print(f"ACCURACY MEDIA (CV): {cv_results['test_accuracy'].mean():.2%}")
print("-" * 50)
print("REPORT DETTAGLIATO PER FASCIA:")
print(classification_report(y_test, y_pred))

print("\n--- CONFUSION MATRIX ---")
conf_matrix = confusion_matrix(y_test, y_pred, labels=LABELS)
conf_df = pd.DataFrame(conf_matrix, index=LABELS, columns=[f"Pred_{l}" for l in LABELS])
print(conf_df)

preprocessor_fitted = final_model.named_steps['preprocessor']
ohe_features = preprocessor_fitted.named_transformers_['cat_encode'].get_feature_names_out()
title_features = [f"title_{f}" for f in preprocessor_fitted.named_transformers_['title_vec'].get_feature_names_out()]
text_features = [f"text_{f}" for f in preprocessor_fitted.named_transformers_['text_vec'].get_feature_names_out()]
num_features = ['hour', 'img_blurriness', 'img_warmth', 'img_saturation', 'img_centering', 'is_weekend']

all_feature_names = np.concatenate([title_features, text_features, ohe_features, num_features])

# Creazione classifica
importances = final_model.named_steps['classifier'].feature_importances_
feat_importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
feat_importance_df = feat_importance_df.sort_values(by='importance', ascending=False)

print("\n--- TOP 10 FEATURE REALI ---")
print(feat_importance_df.head(10))