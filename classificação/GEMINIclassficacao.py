# %% [markdown]
# # Trabalho de Inteligência Artificial - Classificação
# **Alunos:** [Seus Nomes Aqui]
# **Objetivo:** Classificar a gravidade do Parkinson (Leve/Grave) utilizando o dataset 'parkinsons_updrs'.

# %%
# --- 1. Importações (Centralizadas) ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn - Pré-processamento e Seleção
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Sklearn - Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Sklearn - Métricas
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score

# Configuração de plot
plt.style.use('ggplot')

# %% [markdown]
# # 2. Carregamento e Pré-processamento dos Dados
# Carregamos o dataset e transformamos o problema de regressão em classificação binária usando a mediana.

# %%
# Carregar dados
parkinsons = pd.read_csv('./parkinsons+telemonitoring/parkinsons_updrs.data')

# Criar Target Binário (Classificação)
# 1 = Grave (acima da mediana), 0 = Leve (abaixo da mediana)
mediana_updrs = parkinsons['total_UPDRS'].median()
y = (parkinsons['total_UPDRS'] > mediana_updrs).astype(int)

# Remover colunas que não são features preditivas ou são o target original
colunas_remover = ['subject#', 'test_time', 'motor_UPDRS', 'total_UPDRS']
X = parkinsons.drop(colunas_remover, axis=1)

print(f"Shape dos dados: {X.shape}")
print(f"Distribuição das classes:\n{y.value_counts()}")

# Divisão Treino e Teste (Faremos apenas UMA divisão para todos os modelos para comparação justa)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalização (Fit no treino, Transform no teste)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# # 3. Análise Exploratória de Dados (EDA)
# Visualização da correlação entre as variáveis para entender os dados antes da modelagem.

# %%
# Reconstruir DataFrame normalizado apenas para visualização
df_eda = pd.DataFrame(X_train_scaled, columns=X.columns)
df_eda['TARGET'] = y_train.values

plt.figure(figsize=(12, 10))
sns.heatmap(df_eda.corr(), annot=False, cmap='coolwarm')
plt.title("Mapa de Correlação das Features")
plt.show()
# [Adicione aqui uma célula de texto explicando o gráfico acima]

# %% [markdown]
# # 4. Treinamento e Avaliação dos Modelos
# Função auxiliar para treinar, prever e gerar todas as métricas exigidas.

# %%
# DataFrame para armazenar resultados finais
resultados = []

def avaliar_modelo(modelo, nome, X_train, y_train, X_test, y_test):
    # Treino
    modelo.fit(X_train, y_train)
    
    # Predição
    y_pred = modelo.predict(X_test)
    
    # Métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"--- Resultados: {nome} ---")
    print(f"Acurácia: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusão - {nome}')
    plt.ylabel('Real')
    plt.xlabel('Predito')
    plt.show()
    
    # Salvar para comparação final
    resultados.append({
        'Modelo': nome,
        'Acurácia': acc,
        'Precisão': prec,
        'Recall': rec,
        'F1-Score': f1
    })
    return modelo

# %% [markdown]
# ## 4.1 Regressão Logística

# %%
log_reg = LogisticRegression(max_iter=1000, random_state=42)
avaliar_modelo(log_reg, "Regressão Logística", X_train_scaled, y_train, X_test_scaled, y_test)

# %% [markdown]
# ## 4.2 Support Vector Machine (SVM)

# %%
svm_clf = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
avaliar_modelo(svm_clf, "SVM", X_train_scaled, y_train, X_test_scaled, y_test)

# %% [markdown]
# ### Visualização SVM com PCA (Opcional mas recomendado)

# %%
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test_scaled)
plt.figure(figsize=(8,6))
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='viridis', edgecolor='k', alpha=0.7)
plt.title("Visualização PCA (2D) das Classes Reais")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.show()

# %% [markdown]
# ## 4.3 Árvore de Decisão

# %%
# Árvore não exige normalização necessariamente, mas usaremos os dados scaled para manter padrão
tree_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
modelo_tree = avaliar_modelo(tree_clf, "Árvore de Decisão", X_train_scaled, y_train, X_test_scaled, y_test)

plt.figure(figsize=(15, 8))
plot_tree(modelo_tree, feature_names=X.columns, filled=True, rounded=True, class_names=['Leve', 'Grave'])
plt.title("Visualização da Árvore de Decisão")
plt.show()

# %% [markdown]
# ## 4.4 MLP (Rede Neural)

# %%
mlp_clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, activation='relu', solver='adam', random_state=42)
avaliar_modelo(mlp_clf, "MLP", X_train_scaled, y_train, X_test_scaled, y_test)

# %% [markdown]
# # 5. Comparação e Conclusão
# Comparativo final das métricas para definir o melhor modelo.

# %%
df_resultados = pd.DataFrame(resultados).set_index('Modelo')
print(df_resultados)

# Gráfico comparativo
df_resultados.plot(kind='bar', figsize=(10, 6))
plt.title("Comparação de Métricas entre Modelos")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.legend(loc='lower right')
plt.show()

# %% [markdown]
# **Conclusão:**
# [ESCREVA AQUI: Qual modelo foi melhor? Por que? Discuta a complexidade vs performance. O modelo SVM demorou mais? A árvore é mais interpretável? O MLP overfitou?]