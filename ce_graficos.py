import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter

# Configurações estéticas globais para os gráficos
sns.set_theme(style="whitegrid")
palette = "viridis"  # Paleta de cores consistente


def clean_and_normalize_models(series):
    """Limpa e normaliza a coluna 'Técnicas de ML Usadas'."""
    if not pd.api.types.is_string_dtype(series):
         series = series.astype(str)
            
    model_normalization_map = {
        'RF': 'Random Forest (Ensemble)',
        'Random Forest': 'Random Forest (Ensemble)',
        'XGBoost': 'Boosting (Ensemble)',
        'Gradient-boosted trees': 'Boosting (Ensemble)',
        'CatBoost': 'Boosting (Ensemble)',
        'AdaBoost': 'Boosting (Ensemble)',
        'LightGBM': 'Boosting (Ensemble)',
        'SGB': 'Boosting (Ensemble)',
        'Stochastic Gradient Boosting': 'Boosting (Ensemble)',
        'Gradient Boosting Machine (GBM)': 'Boosting (Ensemble)',
        'Gradient Boosting': 'Boosting (Ensemble)',
        'Decision Tree': 'Decision Trees',
        'Decision Trees': 'Decision Trees',
        'DTC': 'Decision Trees',
        'Decision Tree Classifier': 'Decision Trees',
        'J48': 'Decision Trees',
        'ANN': 'Redes Neurais (Clássica)',
        'Neural Networks': 'Redes Neurais (Clássica)',
        'NN': 'Redes Neurais (Clássica)',
        'Redes Neurais Artificiais': 'Redes Neurais (Clássica)',
        'MLP': 'Redes Neurais (Clássica)',
        'Multilayer Perceptron': 'Redes Neurais (Clássica)',
        'Deep Learning': 'Redes Neurais Profundas (DL)',
        'TabNet': 'Redes Neurais Profundas (DL)',
        'LSTM': 'Redes Neurais Profundas (DL)',
        'RNN': 'Redes Neurais Profundas (DL)',
        'Rede Neural Recorrente': 'Redes Neurais Profundas (DL)',
        'BiLSTM': 'Redes Neurais Profundas (DL)',
        'GRU': 'Redes Neurais Profundas (DL)',
        'TimesNet': 'Redes Neurais Profundas (DL)',
        'Transformer': 'Redes Neurais Profundas (DL)',
        '1D CNN': 'Redes Neurais Profundas (DL)',
        'SVM': 'Support Vector Machine (SVM)',
        'Support Vector Machine': 'Support Vector Machine (SVM)',
        'k-NN': 'K-Nearest Neighbors (KNN)',
        'KNN': 'K-Nearest Neighbors (KNN)',
        'k-nearest neighbour': 'K-Nearest Neighbors (KNN)',
        'Regressão Logística': 'Regressão Logística',
        'Logistic Regression': 'Regressão Logística',
        'LR': 'Regressão Logística',
        'Logit': 'Regressão Logística',
        'LogitBoost': 'Boosting (Ensemble)',
        'Naive Bayes': 'Naive Bayes',
        'Naïve Bayes': 'Naive Bayes',
        'Modelos Bayesianos': 'Modelos Bayesianos',
        'Bayesian Networks': 'Modelos Bayesianos',
        'ClusteR': 'Clustering (KNN/Outro)',
        'Linear Discriminant Analysis': 'Modelos Lineares',
        'LDA': 'Modelos Lineares'
    }

    noise = [
        'etc', 'ex:', 'para', 'seleção de features', 'outros', 'Ensemble Methods',
        'SHAP', 'LIME', 'interpretabilidade', 'ELO Rating', 'SHAP', 'XAI', 'ProtoDash'
    ]

    all_models = []
    for cell_text in series.dropna():
        cleaned_text = re.sub(r'[\(\)/;]', ',', cell_text)
        models = cleaned_text.split(',')

        for model in models:
            model_strip = model.strip()
            normalized_model = model_normalization_map.get(
                model_strip, model_strip)

            is_noise = any(n.lower() in normalized_model.lower()
                           for n in noise)
            if normalized_model and len(normalized_model) > 3 and not is_noise:
                all_models.append(normalized_model)

    return pd.Series(all_models).value_counts()


def clean_and_normalize_metrics(series):
    """Limpa e normaliza a coluna 'Métricas de Avaliação'."""
    if not pd.api.types.is_string_dtype(series):
         series = series.astype(str)

    metric_normalization_map = {
        'Accuracy': 'Accuracy (Acurácia)',
        'Overall Accuracy': 'Accuracy (Acurácia)',
        'ACC': 'Accuracy (Acurácia)',
        'OA': 'Accuracy (Acurácia)',
        'RPS': 'Ranked Probability Score (RPS)',
        'Ranked Probability Score': 'Ranked Probability Score (RPS)',
        'Rank Probability Score': 'Ranked Probability Score (RPS)',
        'Brier Score': 'Brier Score',
        'BS': 'Brier Score',
        'Log Loss': 'Log Loss',
        'Log-loss': 'Log Loss',
        'Binary_crossentropy': 'Log Loss',
        'Ignorance score': 'Log Loss',
        'IGN': 'Log Loss',
        'RMSE': 'RMSE (Erro Quadrático)',
        'Rmse': 'RMSE (Erro Quadrático)',
        'Root Mean Squared Error': 'RMSE (Erro Quadrático)',
        'AUC': 'AUC',
        'Area Under Curve': 'AUC',
        'Area under the curve': 'AUC',
        'AUC-ROC': 'AUC',
        'ROI': 'Retorno de Investimento (ROI)',
        'Return-on-Investment': 'Retorno de Investimento (ROI)',
        'F1 Score': 'F1 Score',
        'F1-Score': 'F1 Score',
        'F1 score': 'F1 Score',
        'Weighted F1': 'F1 Score',
        'Precision': 'Precision',
        'Weighted Precision': 'Precision',
        'Recall': 'Recall',
        'Weighted Recall': 'Recall',
        'Specificity': 'Specificity',
        'MCC': 'Matthews Correlation Coefficient (MCC)',
        'Matthews Correlation Coefficient': 'Matthews Correlation Coefficient (MCC)',
        'Confusion Matrix': 'Confusion Matrix',
        'Confusion Matrices': 'Confusion Matrix',
    }

    all_metrics = []
    for cell_text in series.dropna():
        cleaned_text = re.sub(r'[\(\)/;]', ',', cell_text)
        metrics = cleaned_text.split(',')

        for metric in metrics:
            metric_strip = metric.strip()
            normalized_metric = metric_normalization_map.get(
                metric_strip, metric_strip)

            if normalized_metric and len(normalized_metric) > 2:
                found = False
                if metric_strip in metric_normalization_map:
                     all_metrics.append(metric_normalization_map[metric_strip])
                     found = True
                if not found:
                    for key, value in metric_normalization_map.items():
                        if key.lower() in normalized_metric.lower():
                            all_metrics.append(value)
                            found = True
                            break
                if not found and len(normalized_metric) > 3 and 'cit' not in normalized_metric:
                     all_metrics.append(normalized_metric)

    return pd.Series(all_metrics).value_counts()


def clean_and_normalize_limitations(df_limit, df_gaps):
    """Codifica tematicamente as colunas 'Limitações' e 'Lacunas'."""
    if not pd.api.types.is_string_dtype(df_limit):
         df_limit = df_limit.astype(str)
    if not pd.api.types.is_string_dtype(df_gaps):
         df_gaps = df_gaps.astype(str)
            
    theme_map = {
        "Datasets Pequenos / Dados Limitados": r'small|pequenos|tamanho da amostra|sample size|limited data|poucos dados|limited amount of data|data is available',
        "Validação Temporal Inadequada": r'temporal|futuro|past|incorret|k-fold|cross-validation|static|rolling window|train/test split|validation method',
        "Dados Desbalanceados": r'desbalanceados|imbalance|unbalanced',
        "Falta de Interpretabilidade": r'interpretabilidade|xai|caixa preta|black box|shap|lime',
        "Qualidade/Falta de Features": r'qualidade dos dados|features|variáveis|dados limitados|incompletos|feature-set|manual feature engineering|feature selection',
        "Overfitting": r'overfitting|sobreajuste',
        "Fatores Imensuráveis (Sorte, Psicologia)": r'randomness|aleatoriedade|unmeasurable|imensuráveis|psicológicos|mental|sorte|luck|last-minute plays|upsets|rain-interrupted',
        "Complexidade Computacional": r'computational complexity|complexidade computacional|computational power',
        "Ignora Partidas Empatadas": r'tied|empates|draw',
    }
    
    all_text = df_limit.dropna().astype(str) + " " + df_gaps.dropna().astype(str)
    themes_found = []
    
    for text in all_text:
        found_in_article = set() 
        for theme, pattern in theme_map.items():
            if re.search(pattern, text, re.IGNORECASE):
                found_in_article.add(theme)
        themes_found.extend(list(found_in_article))
        
    return pd.Series(themes_found).value_counts()


def plot_horizontal_bar(data, title, filename, xlabel, ylabel, figsize=(10, 8)):
    """Função genérica para criar um gráfico de barras horizontal estético."""
    if data.empty:
        print(f"Skipping '{filename}' because no data was provided.")
        return
        
    df_plot = data.to_frame(name='Frequência').reset_index().rename(
        columns={'index': 'Categoria'})

    plt.figure(figsize=figsize)

    # A linha 'legend=False' FOI REMOVIDA
    barplot = sns.barplot(
        x='Frequência',
        y='Categoria',
        data=df_plot.sort_values('Frequência', ascending=False),
        palette=palette,
        hue='Categoria'
    )

    for p in barplot.patches:
        width = p.get_width()
        plt.text(width + 0.3,
                 p.get_y() + p.get_height() / 2,
                 f'{int(width)}', 
                 va='center')

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    max_freq = data.max()
    if max_freq < 10:
         plt.xlim(0, max_freq + 2)
         plt.xticks(ticks=range(0, int(max_freq) + 3))
    else:
         plt.xlim(0, max_freq * 1.15)
            
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Gráfico salvo: {filename}")


def plot_vertical_bar(data, title, filename, xlabel, ylabel, figsize=(10, 6)):
    """Função genérica para criar um gráfico de barras vertical estético."""
    if data.empty:
        print(f"Skipping '{filename}' because no data was provided.")
        return
            
    df_plot = data.to_frame(name='Frequência').reset_index()
    coluna_categoria = df_plot.columns[0]
    df_plot = df_plot.rename(columns={coluna_categoria: 'Categoria'})

    plt.figure(figsize=figsize)

    # A linha 'legend=False' FOI REMOVIDA
    barplot = sns.barplot(
        x='Categoria',
        y='Frequência',
        data=df_plot.sort_values('Frequência', ascending=False),
        palette=palette,
        hue='Categoria'
    )

    for p in barplot.patches:
        height = p.get_height()
        barplot.annotate(format(height, '.0f'),
                         (p.get_x() + p.get_width() / 2., height),
                         ha = 'center', va = 'center',
                         xytext = (0, 9),
                         textcoords = 'offset points')

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    max_freq = data.max()
    if max_freq < 10:
         plt.ylim(0, max_freq + 2)
         plt.yticks(ticks=range(0, int(max_freq) + 3))
    else:
         plt.ylim(0, max_freq * 1.15)
    
    # Rotação SÓ SE NECESSÁRIO
    if len(data) > 6 or max(len(str(s)) for s in data.index) > 10:
        plt.xticks(rotation=45, ha='right', fontsize=10)
    else:
        plt.xticks(rotation=0, ha='center', fontsize=11)
            
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Gráfico salvo: {filename}")


# --- Execução Principal (VERSÃO FINAL CORRIGIDA) ---
try:
    # Verifique se o nome do seu arquivo CSV está correto aqui
    file_name = "14 - 14.csv" 
    df = pd.read_csv(file_name, encoding="utf-8")
    print(f"Planilha '{file_name}' carregada com sucesso.")

    # --- RQ1 (Panorama - Gráfico 1): Esportes ---
    
    # Assegura que a coluna é string e remove espaços extras
    df['Esporte_Limpo'] = df['Esporte'].astype(str).str.strip()

    # 1. Filtra lixo PRIMEIRO
    mascara_filtro = df['Esporte_Limpo'].str.contains("Não se aplica|indústria", na=False, case=False)
    df_filtrado_rq1 = df[~mascara_filtro].copy()

    # 2. CRIA O MAPA DE NORMALIZAÇÃO (PARA UNIFICAR E ENCURTAR)
    # Este mapa é a correção principal para o seu problema
    sport_normalization_map = {
        # Futebol
        'Futebol': 'Futebol',
        'Soccer': 'Futebol',
        # Basquete
        'Basquete (NBA)': 'Basquete',
        'Basquete': 'Basquete',
        # Tênis
        'Tênis': 'Tênis',
        # Hóquei
        'Ice Hockey (NHL)': 'Hóquei',
        # Futebol Americano
        'Futebol Americano (Sim.)': 'Futebol Americano',
        # Cricket
        'Cricket': 'Cricket',
        # Voleibol
        'Voleibol (SuperLiga Brasileira)': 'Voleibol',
        'Voleibol': 'Voleibol',
        # UFC
        'MMA (UFC)': 'UFC',
        'UFC': 'UFC',
        'Artes Marciais Mistas (MMA)': 'UFC'
    }

    # 3. Aplica o mapa
    # O .get(x, x) significa: "Tente encontrar 'x' no mapa. Se não encontrar, use 'x' (o valor original)."
    df_filtrado_rq1['Esporte_Limpo'] = df_filtrado_rq1['Esporte_Limpo'].apply(lambda x: sport_normalization_map.get(x, x))
    
    # 4. Agora, conta os valores
    sport_counts = df_filtrado_rq1['Esporte_Limpo'].value_counts()
    
    plot_vertical_bar(
        data=sport_counts,
        title="RQ1: Frequência de Esportes Investigados",
        filename="RQ1_Esportes_Frequencia.png",  
        xlabel="Esporte",
        ylabel="Frequência (Nº de Artigos)",
        figsize=(10, 7)
    )

    # --- RQ1 (Panorama - Gráfico 2): Anos ---
    df['Ano_Num'] = pd.to_numeric(df['Ano'], errors='coerce')
    year_counts = df['Ano_Num'].dropna().astype(int).value_counts().sort_index()
    plot_vertical_bar(
        data=year_counts,
        title="RQ1: Evolução das Publicações por Ano",
        filename="RQ1_Publicacoes_por_Ano.png",  
        xlabel="Ano de Publicação",
        ylabel="Frequência (Nos de Artigos)"
    )

    # --- RQ2: Modelos de ML ---
    model_counts = clean_and_normalize_models(df['Técnicas de ML Usadas'])
    plot_horizontal_bar(
        data=model_counts.head(10),  
        title="RQ2: Frequência de Modelos de ML Utilizados (Top 10)",
        filename="RQ2_Modelos_Frequencia.png", 
        xlabel="Frequência (Nº de Artigos)",
        ylabel="Família do Modelo"
    )

    # --- RQ3: Métricas de Avaliação ---
    metric_counts = clean_and_normalize_metrics(df['Métricas de Avaliação'])
    plot_horizontal_bar(
        data=metric_counts.head(10),
        title="RQ3: Frequência de Métricas de Avaliação (Top 10)",
        filename="RQ3_Metricas_Frequencia.png",
        xlabel="Frequência (Nº de Artigos)",
        ylabel="Métrica"
    )

    # --- RQ4: Lacunas e Limitações (Quantitativo) ---
    limitation_counts = clean_and_normalize_limitations(
        df['Limitações'], df['Lacunas'])
    plot_vertical_bar(
        data=limitation_counts,
        title="RQ4: Frequência de Limitações e Lacunas Reportadas",
        filename="RQ4_Limitacoes_Frequencia.png",
        xlabel="Tema da Limitação",
        ylabel="Frequência (Nº de Artigos)",
        figsize=(12, 7)
    )

    print("\nAnálise concluída. 5 arquivos .png foram salvos com os nomes corretos para o LaTeX.")

except FileNotFoundError:
    print(f"Erro: A planilha '{file_name}' não foi encontrada. Verifique se o nome do arquivo está correto.")
except Exception as e:
    print(f"Ocorreu um erro durante a análise: {e}")