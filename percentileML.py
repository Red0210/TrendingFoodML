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
from sklearn.dummy import DummyClassifier

# --- 1. CONFIGURAZIONE PERCENTILE ---
P_LEVEL = 0.50  # 0.75 significa che il top 25% dei post è "Successo"
LABELS = ['Insuccesso', 'Successo']

BASE_PROJECT_PATH = Path.home() / "Desktop" / "scrapingForFood"
FILE_NAME = BASE_PROJECT_PATH / "reddit_food_full_dataset.json"

# Caricamento
df = pd.read_json(FILE_NAME)

# --- 2. PULIZIA E CATEGORIZZAZIONE DINAMICA ---
# 1. Rimuovi righe dove manca lo score (se presenti)
df = df.dropna(subset=['score']) 

# 2. FIX CRITICO: Trasforma i valori nulli (None/NaN) in stringhe vuote
df['title'] = df['title'].fillna('')
df['post_text'] = df['post_text'].fillna('')

# 3. Assicurati che tutto sia trattato come stringa (per sicurezza)
df['title'] = df['title'].astype(str)
df['post_text'] = df['post_text'].astype(str)

# Calcolo della soglia basata sul percentile scelto
THRESHOLD = df['score'].quantile(P_LEVEL)

# Creazione del target binario basato sulla soglia del percentile
df['target_binary'] = (df['score'] > THRESHOLD).astype(int)

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

# --- 5. DEFINIZIONE MODELLO ---
def get_binary_model():
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
        n_estimators=500,        # Aumentato da 300
        max_depth=20,            # Evita che il modello diventi troppo "pignolo" (overfitting)
        random_state=42, 
        n_jobs=-1,
        class_weight='balanced'
        ))
    ])

# --- 6. CROSS-VALIDATION (RF) ---
kf = KFold(n_splits=10, shuffle=True, random_state=42)
print(f"[INFO] Analisi Percentile {P_LEVEL*100}% (Soglia calcolata: {THRESHOLD:.2f} upvotes)")
#cv_results = cross_validate(get_binary_model(), X, y, cv=kf, scoring=['accuracy', 'f1'])
print(f"[INFO] Analisi Percentile {P_LEVEL*100}% (Soglia: {THRESHOLD:.2f} upvotes)")
cv_results = cross_validate(
    get_binary_model(), X, y, cv=kf, 
    scoring=['accuracy', 'f1', 'roc_auc']
)
# --- 7. FIT FINALE ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
final_model = get_binary_model().fit(X_train, y_train)
y_pred = final_model.predict(X_test)

# --- 8. OUTPUT RISULTATI ---
print("\n" + "="*50)
print(f"       PERFORMANCE INDEXES (PERCENTILE {P_LEVEL*100}%)")
print("="*50)
print(f"ACCURACY MEDIA (CV): {cv_results['test_accuracy'].mean():.2%}")
print(f"F1-SCORE MEDIO (CV): {cv_results['test_f1'].mean():.2%}")
print(f"AUC-ROC MEDIO  (CV): {cv_results['test_roc_auc'].mean():.2f}")
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
feat_importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)

print("\n--- TOP 10 FEATURE ---")
print(feat_importance_df.head(10))

# --- 10. CONFRONTO CON DUMMY ---
dummy_clf = DummyClassifier(strategy="most_frequent")
cv_dummy = cross_validate(dummy_clf, X, y, cv=kf, scoring=['accuracy', 'f1'])

print("\n" + "="*50)
print("       CONFRONTO: RANDOM FOREST VS DUMMY")
print("="*50)
print(f"RF Accuracy:    {cv_results['test_accuracy'].mean():.2%}")
print(f"Dummy Accuracy: {cv_dummy['test_accuracy'].mean():.2%}")
improvement = (cv_results['test_accuracy'].mean() - cv_dummy['test_accuracy'].mean())
print(f"MIGLIORAMENTO NETTO: +{improvement:.2%}")

# --- 11. ERROR ANALYSIS ---
class_counts = df['target_binary'].value_counts(normalize=True)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
from sklearn.metrics import roc_auc_score
y_probs = final_model.predict_proba(X_test)[:, 1] # Probabilità per la classe "Successo"
auc_finale = roc_auc_score(y_test, y_probs)
print("\n--- ERROR ANALYSIS & VARIANCE ---")
print(f"Distribuzione Reale - Insuccesso: {class_counts[0]:.2%}, Successo: {class_counts[1]:.2%}")
print(f"False Positive Rate (FPR): {fpr:.2%}")
print(f"False Negative Rate (FNR): {fnr:.2%}")
print(f"Varianza Accuracy (CV): {cv_results['test_accuracy'].var():.6f}")
print(f"Deviazione Standard (CV): {cv_results['test_accuracy'].std():.2%}")
print(f"AUC-ROC Finale (Test Set): {auc_finale:.4f}")





def save_results_to_latex(params, metrics, filename="runsResults.tex"):
    """Appende i risultati del Random Forest al file LaTeX inclusa la Std Dev."""
    with open(filename, "a", encoding="utf-8") as f:
        f.write("\n% --- Nuova Run Random Forest --- \n")
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\begin{tabular}{|l|l|}\n\\hline\n")
        
        # Parametri
        f.write(f"Modello & Random Forest \\\\ \\hline\n")
        f.write(f"Numero di Fold & {params['n_splits']} \\\\ \\hline\n")
        f.write(f"Numero Alberi & {params['n_estimators']} \\\\ \\hline\n")
        f.write(f"Profondita Max & {params['max_depth']} \\\\ \\hline\n")
        
        # Risultati metriche
        f.write(f"Accuracy (CV) & {metrics['accuracy']:.2%} \\\\ \\hline\n")
        f.write(f"Std Deviation (CV) & {metrics['std_dev']:.2%} \\\\ \\hline\n") # <--- AGGIUNTO
        f.write(f"F1-Score (CV) & {metrics['f1']:.2%} \\\\ \\hline\n")
        f.write(f"AUC-ROC (CV) & {metrics['auc']:.3f} \\\\ \\hline\n")
        f.write(f"FPR / FNR & {metrics['fpr']:.2%} / {metrics['fnr']:.2%} \\\\ \\hline\n")
        
        f.write("\\hline\n\\end{tabular}\n")
        f.write(f"\\caption{{Risultati RF: Acc {metrics['accuracy']:.2%} ($\\pm$ {metrics['std_dev']:.2%})}}\n")
        f.write("\\end{table}\n")
        f.write("% ---------------- \n")
    print(f"[LATEX] Risultati RF (con Std Dev) salvati in {filename}")

# --- 12. ESPORTAZIONE LATEX ---

parametri_rf = {
    'n_splits': kf.get_n_splits(),
    'n_estimators': 500,  # Ricordati di mettere il valore che usi realmente
    'max_depth': 20      
}

metriche_rf = {
    'accuracy': cv_results['test_accuracy'].mean(),
    'std_dev': cv_results['test_accuracy'].std(), # <--- AGGIUNTO
    'f1': cv_results['test_f1'].mean(),
    'auc': cv_results['test_roc_auc'].mean(),
    'fpr': fpr,
    'fnr': fnr
}

save_results_to_latex(parametri_rf, metriche_rf)