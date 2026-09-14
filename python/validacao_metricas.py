import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 1. Carregamento dos dados
caminho_fake = os.path.join("dados", "fakes.csv")
caminho_true = os.path.join("dados", "true.csv")

try:
    df_fake = pd.read_csv(caminho_fake, encoding='utf-8')
    df_true = pd.read_csv(caminho_true, encoding='utf-8')
except UnicodeDecodeError:
    df_fake = pd.read_csv(caminho_fake, encoding='latin-1')
    df_true = pd.read_csv(caminho_true, encoding='latin-1')

df_fake['label'] = 'FAKE'
df_true['label'] = 'REAL'

df = pd.concat([df_fake, df_true], ignore_index=True)

colunas_texto = [c for c in df.columns if c.lower() in ['text', 'texto', 'noticia', 'conteudo', 'body', 'title', 'titulo']]
col_texto = colunas_texto[0] if colunas_texto else df.columns[0]
df = df.dropna(subset=[col_texto]).copy()

# 2. Split estratificado (mantém a proporção exata de Fake e Real no teste)
X_train, X_test, y_train, y_test = train_test_split(
    df[col_texto].astype(str),
    df['label'],
    test_size=0.20,
    random_state=42,
    stratify=df['label']
)

# 3. TF-IDF com n-grams
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Modelo com compensação de desbalanceamento (class_weight='balanced')
modelo = LogisticRegression(class_weight='balanced', max_iter=1000)
modelo.fit(X_train_vec, y_train)

# 5. Avaliação Formal solicitada pelo orientador
y_pred = modelo.predict(X_test_vec)

print("=" * 65)
print("RELATÓRIO DE MÉTRICAS DETALHADO (PRECISÃO, RECALL, F1-SCORE)")
print("=" * 65)
print(classification_report(y_test, y_pred, digits=4))

# 6. Matriz de Confusão
cm = confusion_matrix(y_test, y_pred, labels=['REAL', 'FAKE'])

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predito REAL', 'Predito FAKE'], 
            yticklabels=['Verdadeiro REAL', 'Verdadeiro FAKE'])
plt.title("Matriz de Confusão (Baseline)")
plt.ylabel("Rótulo Real")
plt.xlabel("Rótulo Predito")
plt.tight_layout()
plt.show()