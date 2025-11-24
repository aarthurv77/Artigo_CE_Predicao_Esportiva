import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configurações Estéticas ---
sns.set_theme(style="whitegrid")
palette = "viridis"

def plot_horizontal(data_dict, title, filename, xlabel, ylabel):
    """Gera gráfico de barras horizontal a partir de um dicionário."""
    # Converte dicionário para DataFrame
    df = pd.DataFrame(list(data_dict.items()), columns=['Categoria', 'Frequência'])
    df = df.sort_values('Frequência', ascending=False)

    plt.figure(figsize=(10, 8))
    barplot = sns.barplot(
        x='Frequência',
        y='Categoria',
        data=df,
        palette=palette,
        hue='Categoria',
        legend=False
    )

    # Adiciona números nas barras
    for p in barplot.patches:
        width = p.get_width()
        plt.text(width + 0.2, p.get_y() + p.get_height() / 2, 
                 f'{int(width)}', va='center')

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlim(0, df['Frequência'].max() * 1.15)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Salvo: {filename}")

def plot_vertical(data_dict, title, filename, xlabel, ylabel, sort_by_key=False):
    """Gera gráfico de barras vertical a partir de um dicionário."""
    df = pd.DataFrame(list(data_dict.items()), columns=['Categoria', 'Frequência'])
    
    if sort_by_key:
        df = df.sort_values('Categoria') # Para Anos
    else:
        df = df.sort_values('Frequência', ascending=False) # Para outros

    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(
        x='Categoria',
        y='Frequência',
        data=df,
        palette=palette,
        hue='Categoria',
        legend=False
    )

    # Adiciona números nas barras
    for p in barplot.patches:
        height = p.get_height()
        barplot.annotate(f'{int(height)}',
                         (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='center',
                         xytext=(0, 5), textcoords='offset points')

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.ylim(0, df['Frequência'].max() * 1.15)
    
    # Rotaciona rótulos se houver muitos itens
    if len(df) > 5:
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Salvo: {filename}")

# ==========================================
# DADOS EXATOS (HARDCODED)
# ==========================================

# 1. RQ1 - Esportes
dados_esportes = {
    'Futebol': 5,
    'Basquete': 3,
    'Tênis': 2,
    'Hóquei': 1, # Encurtado de Hóquei no Gelo para caber melhor
    'UFC': 1,
    'Cricket': 1,
    'Voleibol': 1
}

# 2. RQ1 - Anos
dados_anos = {
    2014: 1,
    2021: 2,
    2022: 3,
    2023: 3,
    2024: 4,
    2025: 1
}

# 3. RQ2 - Modelos
dados_modelos = {
    'Boosting (Ensemble)': 8,
    'Redes Neurais (Clássica)': 8, # Empatado
    'Random Forest (Ensemble)': 8, # Empatado
    'SVM': 6,
    'Regressão Logística': 6,
    'KNN': 5,
    'Decision Trees': 5,
    'Redes Neurais Profundas (DL)': 4,
    'Naive Bayes': 2,
    'Modelos Bayesianos': 1,
    'Clustering': 1,
    'Modelos Lineares': 1
}

# 4. RQ3 - Métricas
dados_metricas = {
    'Accuracy (Acurácia)': 10,
    'Precision': 5,
    'Recall': 5,
    'F1 Score': 5,
    'AUC': 4,
    'RMSE': 3,
    'RPS (Ranked Prob. Score)': 3,
    'Log Loss': 3,
    'Brier Score': 3,
    'ROI': 2,
    'Confusion Matrix': 1,
    'Specificity': 1,
    'MCC': 1
}

# 5. RQ4 - Limitações
dados_limitacoes = {
    'Falta de Features': 11,
    'Complexidade': 3,
    'Datasets Pequenos': 2,
    'Fatores Imensuráveis': 2,
    'Interpretabilidade': 2
}

# ==========================================
# GERAÇÃO DOS GRÁFICOS
# ==========================================
print("Gerando gráficos com valores exatos...")

# RQ1 Esportes (Vertical)
plot_vertical(dados_esportes, "RQ1: Frequência de Esportes Investigados", 
              "RQ1_Esportes_Frequencia.png", "Esporte", "Frequência")

# RQ1 Anos (Vertical - Ordenado por Ano)
plot_vertical(dados_anos, "RQ1: Evolução das Publicações por Ano", 
              "RQ1_Publicacoes_por_Ano.png", "Ano", "Frequência", sort_by_key=True)

# RQ2 Modelos (Horizontal)
plot_horizontal(dados_modelos, "RQ2: Frequência de Modelos de ML Utilizados", 
                "RQ2_Modelos_Frequencia.png", "Frequência", "Família do Modelo")

# RQ3 Métricas (Horizontal)
plot_horizontal(dados_metricas, "RQ3: Frequência de Métricas de Avaliação", 
                "RQ3_Metricas_Frequencia.png", "Frequência", "Métrica")

# RQ4 Limitações (Vertical)
plot_vertical(dados_limitacoes, "RQ4: Principais Desafios e Limitações", 
              "RQ4_Limitacoes_Frequencia.png", "Desafio", "Frequência")

print("Sucesso! Gráficos gerados.")