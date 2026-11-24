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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- 1. CONFIGURAZIONE BINARIA ---
THRESHOLD = 500  # Sopra i 500 upvote è "Successo"
LABELS = ['Insuccesso', 'Successo']

BASE_PROJECT_PATH = Path.home() / "Desktop" / "scrapingForFood"
FILE_NAME = BASE_PROJECT_PATH / "reddit_food_full_dataset.json"

# Caricamento
df = pd.read_json(FILE_NAME)

# --- 2. PULIZIA E CATEGORIZZAZIONE BINARIA ---
df = df.dropna(subset=['score']) 

# Creazione del target binario: 0 se <= THRESHOLD, 1 se > THRESHOLD
df['target_binary'] = (df['score'] > THRESHOLD).astype(int)
# Mappatura nomi per il report
target_names = LABELS 

# Definizione Feature e Target
X = df[['hour', 'img_blurriness', 'img_warmth', 'img_saturation', 
        'img_centering', 'is_weekend', 'day_of_week', 'title', 'post_text']]
y = df['target_binary']

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

# --- 5. DEFINIZIONE MODELLO (CON BILANCIAMENTO) ---
def get_binary_model():
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=300, 
            random_state=42, 
            n_jobs=-1,
            class_weight='balanced' # Bilancia Insuccesso vs Successo
        ))
    ])

# --- 6. CROSS-VALIDATION ---
kf = KFold(n_splits=10, shuffle=True, random_state=42)
print(f"[INFO] Valutazione Classificazione Binaria (Soglia: {THRESHOLD} upvotes)...")
cv_results = cross_validate(get_binary_model(), X, y, cv=kf, scoring=['accuracy', 'f1'])

# --- 7. FIT FINALE E ANALISI DETTAGLIATA ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
final_model = get_binary_model().fit(X_train, y_train)
y_pred = final_model.predict(X_test)

# --- 8. OUTPUT RISULTATI ---
print("\n" + "="*50)
print("       PERFORMANCE INDEXES: BINARY MODEL")
print("="*50)
print(f"ACCURACY MEDIA (CV): {cv_results['test_accuracy'].mean():.2%}")
print(f"F1-SCORE MEDIO (CV): {cv_results['test_f1'].mean():.2%}")
print("-" * 50)
print("REPORT DETTAGLIATO:")
print(classification_report(y_test, y_pred, target_names=LABELS))

print("\n--- CONFUSION MATRIX ---")
conf_matrix = confusion_matrix(y_test, y_pred)
conf_df = pd.DataFrame(conf_matrix, index=[f"Reale_{l}" for l in LABELS], 
                     columns=[f"Pred_{l}" for l in LABELS])
print(conf_df)

# --- 9. FEATURE IMPORTANCE ---
preprocessor_fitted = final_model.named_steps['preprocessor']
ohe_features = preprocessor_fitted.named_transformers_['cat_encode'].get_feature_names_out()
title_features = [f"title_{f}" for f in preprocessor_fitted.named_transformers_['title_vec'].get_feature_names_out()]
text_features = [f"text_{f}" for f in preprocessor_fitted.named_transformers_['text_vec'].get_feature_names_out()]
num_features = ['hour', 'img_blurriness', 'img_warmth', 'img_saturation', 'img_centering', 'is_weekend']

all_feature_names = np.concatenate([title_features, text_features, ohe_features, num_features])
importances = final_model.named_steps['classifier'].feature_importances_

feat_importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
feat_importance_df = feat_importance_df.sort_values(by='importance', ascending=False)

print("\n--- TOP 10 FEATURE PER IL SUCCESSO ---")
print(feat_importance_df.head(10))

from sklearn.dummy import DummyClassifier

# 1. Creiamo il classificatore "stupido" (strategia basata sulla classe più frequente)
dummy_clf = DummyClassifier(strategy="most_frequent")

# 2. Valutazione con Cross-Validation (stesso metodo usato per il Random Forest)
cv_dummy = cross_validate(dummy_clf, X, y, cv=kf, scoring=['accuracy', 'f1'])

print("\n" + "="*50)
print("       CONFRONTO: RANDOM FOREST VS DUMMY")
print("="*50)
print(f"RF Accuracy:    {cv_results['test_accuracy'].mean():.2%}")
print(f"Dummy Accuracy: {cv_dummy['test_accuracy'].mean():.2%}")
print("-" * 50)
improvement = (cv_results['test_accuracy'].mean() - cv_dummy['test_accuracy'].mean())
print(f"MIGLIORAMENTO NETTO: +{improvement:.2%}")

from sklearn.metrics import confusion_matrix

# 1. Calcolo della frequenza delle classi (Dataset reale)
class_counts = df['target_binary'].value_counts(normalize=True)
print("\n--- FREQUENZA REALE DELLE CLASSI ---")
print(f"Insuccesso (0): {class_counts[0]:.2%}")
print(f"Successo (1):   {class_counts[1]:.2%}")

# 2. Estrazione FPR e FNR dalla Confusion Matrix
# tn = True Negatives, fp = False Positives, fn = False Negatives, tp = True Positives
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

fpr = fp / (fp + tn) # False Positive Rate
fnr = fn / (fn + tp) # False Negative Rate

print("\n--- ERROR ANALYSIS (Rates) ---")
print(f"False Positive Rate (FPR): {fpr:.2%} (Post falliti scambiati per successi)")
print(f"False Negative Rate (FNR): {fnr:.2%} (Post di successo non visti dal modello)")

# 3. Nota sulla Varianza
# La varianza si ottiene guardando le performance tra i diversi fold della Cross-Validation
print("\n--- STABILITÀ DEL MODELLO (Varianza CV) ---")
print(f"Varianza Accuracy: {cv_results['test_accuracy'].var():.6f}")
print(f"Deviazione Standard: {cv_results['test_accuracy'].std():.2%}")