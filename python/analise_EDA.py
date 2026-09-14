import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar os dois CSVs
caminho_fake = os.path.join("dados", "fakes.csv")
caminho_true = os.path.join("dados", "true.csv")

try:
    df_fake = pd.read_csv(caminho_fake, encoding='utf-8')
    df_true = pd.read_csv(caminho_true, encoding='utf-8')
except UnicodeDecodeError:
    df_fake = pd.read_csv(caminho_fake, encoding='latin-1')
    df_true = pd.read_csv(caminho_true, encoding='latin-1')

# 2. Rotular e Concatenar
df_fake['label'] = 'FAKE'
df_true['label'] = 'REAL'

df = pd.concat([df_fake, df_true], ignore_index=True)

# Embaralhar as linhas
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 3. Informações gerais
print("=" * 60)
print(f"Total FAKE: {len(df_fake)} | Total REAL: {len(df_true)}")
print(f"Total Geral: {df.shape[0]} linhas x {df.shape[1]} colunas")
print("=" * 60)

print("\nColunas encontradas:")
print(df.columns.tolist())

print("\nPrimeiras 3 linhas:")
print(df.head(3))

print("\nValores nulos por coluna:")
print(df.isnull().sum())

# 4. Detecção da coluna de texto e cálculo do tamanho
colunas_texto = [c for c in df.columns if c.lower() in ['text', 'texto', 'noticia', 'conteudo', 'body']]
col_texto = colunas_texto[0] if colunas_texto else df.columns[0]
print(f"\nColuna selecionada para análise textual: '{col_texto}'")

df_limpo = df.dropna(subset=[col_texto]).copy()
df_limpo['word_count'] = df_limpo[col_texto].astype(str).apply(lambda x: len(x.split()))

print("\nEstatísticas do tamanho das notícias (palavras):")
print(df_limpo.groupby('label')['word_count'].describe())

# 5. Gráficos
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.countplot(data=df_limpo, x='label', palette='viridis')
plt.title("Distribuição das Classes (Fake vs Real)")
plt.xlabel("Classe")
plt.ylabel("Quantidade")

plt.subplot(1, 2, 2)
sns.boxplot(data=df_limpo, x='label', y='word_count', palette='viridis', showfliers=False)
plt.title("Contagem de Palavras por Classe")
plt.xlabel("Classe")
plt.ylabel("Palavras")

plt.tight_layout()
plt.show()