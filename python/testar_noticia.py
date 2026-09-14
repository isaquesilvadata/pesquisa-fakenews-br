import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Carregar os CSVs
print("Carregando arquivos...")
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

# 2. Identificar a coluna de texto
colunas_texto = [c for c in df.columns if c.lower() in ['text', 'texto', 'noticia', 'conteudo', 'body', 'title', 'titulo']]
col_texto = colunas_texto[0] if colunas_texto else df.columns[0]
print(f"Treinando baseline com a coluna: '{col_texto}'")

# Remover nulos
df = df.dropna(subset=[col_texto]).copy()

# 3. Divisão Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(
    df[col_texto].astype(str), 
    df['label'], 
    test_size=0.2, 
    random_state=42, 
    stratify=df['label']
)

# 4. Vetorização (TF-IDF focado em português)
print("Construindo matriz TF-IDF...")
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Treinamento da baseline (com class_weight='balanced' devido ao desbalanceamento)
print("Treinando classificador linear...")
modelo = LogisticRegression(class_weight='balanced', max_iter=1000)
modelo.fit(X_train_vec, y_train)

# Avaliação rápida
preds = modelo.predict(X_test_vec)
print(f"\nAcurácia no conjunto de teste: {accuracy_score(y_test, preds):.2%}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, preds))

# 6. Loop Interativo de Input no Terminal
print("=" * 60)
print("MODO DE TESTE INTERATIVO (Digite 'sair' para encerrar)")
print("=" * 60)

while True:
    noticia_input = input("\nCole a notícia ou manchete para testar: ")
    
    if noticia_input.strip().lower() in ['sair', 'exit', 'quit']:
        print("Encerrando...")
        break
        
    if not noticia_input.strip():
        continue
        
    # Vetorizar e predizer
    vec_input = vectorizer.transform([noticia_input])
    pred = modelo.predict(vec_input)[0]
    prob = modelo.predict_proba(vec_input)[0]
    classes = modelo.classes_
    
    idx_classe = list(classes).index(pred)
    confianca = prob[idx_classe] * 100
    
    print("\n--- RESULTADO DA PREDIÇÃO ---")
    print(f"Classificação: {pred}")
    print(f"Confiança: {confianca:.2f}%")
    print("-" * 30)