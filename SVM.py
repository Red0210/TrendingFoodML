import pandas as pd
import numpy as np
import re
from pathlib import Path
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import cross_validate, KFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.dummy import DummyClassifier

# --- 1. CONFIGURAZIONE PERCENTILE ---
P_LEVEL = 0.50  
LABELS = ['Insuccesso', 'Successo']

BASE_PROJECT_PATH = Path.home() / "Desktop" / "scrapingForFood"
FILE_NAME = BASE_PROJECT_PATH / "reddit_food_full_dataset.json"

# Caricamento
df = pd.read_json(FILE_NAME)

# --- 2. PULIZIA E CATEGORIZZAZIONE DINAMICA ---
df = df.dropna(subset=['score']) 

df['title'] = df['title'].fillna('')
df['post_text'] = df['post_text'].fillna('')
df['title'] = df['title'].astype(str)
df['post_text'] = df['post_text'].astype(str)

THRESHOLD = df['score'].quantile(P_LEVEL)
df['target_binary'] = (df['score'] > THRESHOLD).astype(int)

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

# --- 4. PIPELINE DI PREPROCESSING (CON SCALING PER SVM) ---
preprocessor = ColumnTransformer(
    transformers=[
        ('title_vec', TfidfVectorizer(tokenizer=advanced_tokenizer, max_features=150, token_pattern=None), 'title'),
        ('text_vec', TfidfVectorizer(tokenizer=advanced_tokenizer, max_features=150, token_pattern=None), 'post_text'),
        ('cat_encode', OneHotEncoder(handle_unknown='ignore'), ['day_of_week']),
        # SVM richiede obbligatoriamente lo scaling delle feature numeriche
        ('num_features', StandardScaler(), ['hour', 'img_blurriness', 'img_warmth', 'img_saturation', 'img_centering', 'is_weekend'])
    ]
)

# --- 5. DEFINIZIONE MODELLO SVM ---
def get_binary_model():
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', SVC(
            kernel='rbf',           # Kernel non lineare 'linear' 'rbf'(standard) 'poly' 'classifier__degree': [3]
            C=1.0,                  # Regolarizzazione
            probability=True,       # Necessario per AUC-ROC e predict_proba
            class_weight='balanced',# Bilancia le classi automaticamente
            random_state=42
        ))
    ])

# --- 6. CROSS-VALIDATION (SVM) ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
print(f"[INFO] Analisi Percentile {P_LEVEL*100}% (Soglia: {THRESHOLD:.2f} upvotes)")

cv_results = cross_validate(
    get_binary_model(), X, y, cv=kf, 
    scoring=['accuracy', 'f1', 'roc_auc'],
    n_jobs=-1 # Usa più core per velocizzare (SVM è più lenta di RF)
)

# --- 7. FIT FINALE ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
final_model = get_binary_model().fit(X_train, y_train)
y_pred = final_model.predict(X_test)

# --- 8. OUTPUT RISULTATI ---
print("\n" + "="*50)
print(f"        PERFORMANCE INDEXES (SVM - PERCENTILE {P_LEVEL*100}%)")
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

# --- 9. CONFRONTO CON DUMMY ---
dummy_clf = DummyClassifier(strategy="most_frequent")
cv_dummy = cross_validate(dummy_clf, X, y, cv=kf, scoring=['accuracy', 'f1'])

print("\n" + "="*50)
print("        CONFRONTO: SVM VS DUMMY")
print("="*50)
print(f"SVM Accuracy:   {cv_results['test_accuracy'].mean():.2%}")
print(f"Dummy Accuracy: {cv_dummy['test_accuracy'].mean():.2%}")
improvement = (cv_results['test_accuracy'].mean() - cv_dummy['test_accuracy'].mean())
print(f"MIGLIORAMENTO NETTO: +{improvement:.2%}")

# --- 10. ERROR ANALYSIS ---
class_counts = df['target_binary'].value_counts(normalize=True)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

y_probs = final_model.predict_proba(X_test)[:, 1] 
auc_finale = roc_auc_score(y_test, y_probs)

print("\n--- ERROR ANALYSIS & VARIANCE ---")
print(f"Distribuzione Reale - Insuccesso: {class_counts[0]:.2%}, Successo: {class_counts[1]:.2%}")
print(f"False Positive Rate (FPR): {fpr:.2%}")
print(f"False Negative Rate (FNR): {fnr:.2%}")
print(f"Deviazione Standard Accuracy (CV): {cv_results['test_accuracy'].std():.2%}")
print(f"AUC-ROC Finale (Test Set): {auc_finale:.4f}")


# --- ESEMPIO DI INTEGRAZIONE DOPO IL FIT ---
def save_results_to_latex(params, metrics, filename="runsResults.tex"):
    """
    Appende i risultati SVM al file runsResults.tex in formato tabella LaTeX.
    Include la Deviazione Standard per la valutazione della stabilità.
    """
    with open(filename, "a", encoding="utf-8") as f:
        f.write("\n% --- Nuova Run SVM --- \n")
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\begin{tabular}{|l|l|}\n\\hline\n")
        
        # Righe dei parametri configurazione
        f.write(f"Numero di Fold & {params['n_splits']} \\\\ \\hline\n")
        f.write(f"Kernel utilizzato & {params['kernel']} \\\\ \\hline\n")
        f.write(f"Valore Parametro C & {params['C']} \\\\ \\hline\n")
        
        # Righe dei risultati metriche con Standard Deviation
        f.write(f"Accuracy (CV) & {metrics['accuracy']:.2%} \\\\ \\hline\n")
        f.write(f"Std Deviation (CV) & {metrics['std_dev']:.2%} \\\\ \\hline\n") # <--- AGGIUNTO
        f.write(f"F1-Score (CV) & {metrics['f1']:.2%} \\\\ \\hline\n")
        f.write(f"AUC-ROC (CV) & {metrics['auc']:.3f} \\\\ \\hline\n")
        f.write(f"FPR / FNR & {metrics['fpr']:.2%} / {metrics['fnr']:.2%} \\\\ \\hline\n")
        
        f.write("\\hline\n\\end{tabular}\n")
        # Caption con l'intervallo di confidenza
        f.write(f"\\caption{{Risultati SVM: Acc {metrics['accuracy']:.2%} ($\\pm$ {metrics['std_dev']:.2%}) con Kernel {params['kernel']} e C={params['C']}}}\n")
        f.write("\\end{table}\n")
        f.write("% ---------------- \n")

    print(f"[LATEX] Tabella SVM (con Std Dev) aggiunta a {filename}")


# 1. Prepara i parametri (usa i valori reali del tuo modello)
# Definizione dei parametri e delle metriche per la SVM
current_params = {
    'n_splits': kf.get_n_splits(),
    'kernel': 'rbf',  # Inserisci il kernel che stai usando
    'C': 1.0          # Inserisci il valore di C che stai usando
}

current_metrics = {
    'accuracy': cv_results['test_accuracy'].mean(),
    'std_dev': cv_results['test_accuracy'].std(), # <--- Calcolo della Standard Deviation
    'f1': cv_results['test_f1'].mean(),
    'auc': cv_results['test_roc_auc'].mean(),
    'fpr': fpr, 
    'fnr': fnr
}

# Chiamata del salvataggio
save_results_to_latex(current_params, current_metrics)