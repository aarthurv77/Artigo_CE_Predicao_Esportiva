import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter

# Configurações estéticas globais para os gráficos
sns.set_theme(style="whitegrid")
palette = "viridis" # Paleta de cores consistente

def clean_and_normalize_models(series):
    """Limpa e normaliza a coluna 'Técnicas de ML Usadas'."""
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
        'Decision Tree': 'Decision Trees',
        'Decision Trees': 'Decision Trees',
        'DTC': 'Decision Trees',
        'Decision Tree Classifier': 'Decision Trees',
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
        'SVM': 'Support Vector Machine (SVM)',
        'Support Vector Machine': 'Support Vector Machine (SVM)',
        'k-NN': 'K-Nearest Neighbors (KNN)',
        'KNN': 'K-Nearest Neighbors (KNN)',
        'Regressão Logística': 'Regressão Logística',
        'Logistic Regression': 'Regressão Logística',
        'LR': 'Regressão Logística',
        'Logit': 'Regressão Logística',
        'Naive Bayes': 'Naive Bayes',
        'Naïve Bayes': 'Naive Bayes',
        'Modelos Bayesianos': 'Modelos Bayesianos',
        'Bayesian Networks': 'Modelos Bayesianos',
    }
    
    noise = [
        'etc', 'ex:', 'para', 'seleção de features', 'outros', 'Ensemble Methods', 
        'SHAP', 'LIME', 'interpretabilidade', 'ELO Rating', 'SHAP'
    ]
    
    all_models = []
    for cell_text in series.dropna():
        # Substitui parênteses e caracteres especiais por vírgulas para separar
        cleaned_text = re.sub(r'[\(\)/;]', ',', cell_text)
        models = cleaned_text.split(',')
        
        for model in models:
            model_strip = model.strip()
            
            # Normaliza
            normalized_model = model_normalization_map.get(model_strip, model_strip)
            
            # Filtra ruídos e nomes curtos/genéricos
            is_noise = any(n.lower() in normalized_model.lower() for n in noise)
            if normalized_model and len(normalized_model) > 3 and not is_noise:
                all_models.append(normalized_model)
                
    return pd.Series(all_models).value_counts()

def clean_and_normalize_metrics(series):
    """Limpa e normaliza a coluna 'Métricas de Avaliação'."""
    metric_normalization_map = {
        'Accuracy': 'Accuracy (Acurácia)',
        'Overall Accuracy': 'Accuracy (Acurácia)',
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
        'ROI': 'Retorno de Investimento (ROI)',
        'Return-on-Investment': 'Retorno de Investimento (ROI)',
        'F1 Score': 'F1 Score',
        'F1 score': 'F1 Score',
        'Precision': 'Precision',
        'Recall': 'Recall',
        'Sensitivity': 'Recall',
        'Specificity': 'Specificity',
    }
    
    all_metrics = []
    for cell_text in series.dropna():
        cleaned_text = re.sub(r'[\(\)/;]', ',', cell_text)
        metrics = cleaned_text.split(',')
        
        for metric in metrics:
            metric_strip = metric.strip()
            normalized_metric = metric_normalization_map.get(metric_strip, metric_strip)
            
            if normalized_metric and len(normalized_metric) > 2:
                # Tratamento especial para evitar contagem de métricas parciais
                found = False
                for key, value in metric_normalization_map.items():
                    if key.lower() in normalized_metric.lower():
                        all_metrics.append(value)
                        found = True
                        break
                if not found and len(normalized_metric) > 3 and 'cit' not in normalized_metric:
                     all_metrics.append(normalized_metric)

    # Remove duplicatas por artigo (um artigo pode citar 'Accuracy' 3x)
    return pd.Series(all_metrics).value_counts()

def clean_and_normalize_limitations(df_limit, df_gaps):
    """Codifica tematicamente as colunas 'Limitações' e 'Lacunas'."""
    theme_map = {
        "Datasets Pequenos": r'small|pequenos|tamanho da amostra|sample size|limited data|poucos dados',
        "Validação Temporal Inadequada": r'temporal|futuro|past|incorret|k-fold|cross-validation',
        "Dados Desbalanceados": r'desbalanceados|imbalance|unbalanced',
        "Falta de Interpretabilidade": r'interpretabilidade|xai|caixa preta|black box|shap|lime',
        "Qualidade/Falta de Features": r'qualidade dos dados|features|variáveis|dados limitados|incompletos',
        "Overfitting": r'overfitting|sobreajuste',
    }
    
    all_text = df_limit.dropna().astype(str) + " " + df_gaps.dropna().astype(str)
    themes_found = []
    
    for text in all_text:
        found_in_article = set() # Evita contagem dupla do mesmo tema no *mesmo* artigo
        for theme, pattern in theme_map.items():
            if re.search(pattern, text, re.IGNORECASE):
                found_in_article.add(theme)
        themes_found.extend(list(found_in_article))
        
    return pd.Series(themes_found).value_counts()

def plot_horizontal_bar(data, title, filename, xlabel, ylabel, figsize=(10, 8)):
    """Função genérica para criar um gráfico de barras horizontal estético."""
    # Esta linha é crucial
    df_plot = data.to_frame(name='Frequência').reset_index().rename(columns={'index': 'Categoria'})
    
    plt.figure(figsize=figsize)
    
    # Adicionado 'hue' e 'legend=False' para resolver o FutureWarning
    barplot = sns.barplot(
        x='Frequência',
        y='Categoria',
        data=df_plot.sort_values('Frequência', ascending=False),
        palette=palette,
        hue='Categoria',
        legend=False
    )
    
    # Adiciona rótulos de dados
    for p in barplot.patches:
        width = p.get_width()
        plt.text(width + 0.3, # Posição x (um pouco à frente da barra)
                 p.get_y() + p.get_height() / 2, # Posição y (centro da barra)
                 f'{width:.0f}', # O texto
                 va='center')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlim(0, data.max() * 1.15) # Ajusta o limite de x para dar espaço aos rótulos
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Gráfico salvo: {filename}")

def plot_vertical_bar(data, title, filename, xlabel, ylabel, figsize=(10, 6)):
    """Função genérica para criar um gráfico de barras vertical estético."""
    
    df_plot = data.to_frame(name='Frequência').reset_index()
    
    coluna_categoria = df_plot.columns[0]
    df_plot = df_plot.rename(columns={coluna_categoria: 'Categoria'})

    plt.figure(figsize=figsize)
    
    # Agora o 'x' e 'hue' funcionarão, pois a coluna 'Categoria' sempre existirá
    barplot = sns.barplot(
        x='Categoria',
        y='Frequência',
        data=df_plot.sort_values('Frequência', ascending=False),
        palette=palette,
        hue='Categoria',
        legend=False
    )
    
    # Adiciona rótulos de dados
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.0f'),
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha = 'center', va = 'center',
                         xytext = (0, 9),
                         textcoords = 'offset points')

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.ylim(0, data.max() * 1.15) # Ajusta o limite de y para dar espaço aos rótulos
    
    # Ajuste para rotação de rótulos se houver muitas categorias (ex: RQ4)
    if len(data) > 3 or max(len(str(s)) for s in data.index) > 15: 
        plt.xticks(rotation=45, ha='right', fontsize=10) 
    else:
        plt.xticks(rotation=0, ha='center', fontsize=11) 
        
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Gráfico salvo: {filename}")

# --- Execução Principal ---
try:
    df = pd.read_csv("14 - 14.csv")
    print("Planilha '14 - 14.csv' carregada com sucesso.")

    # --- RQ1: Modelos de ML ---
    model_counts = clean_and_normalize_models(df['Técnicas de ML Usadas'])
    plot_horizontal_bar(
        data=model_counts.head(10), # Mostra o Top 10
        title="RQ1: Frequência de Modelos de ML Utilizados (Top 10)",
        filename="RQ1_Modelos_Frequencia.png",
        xlabel="Frequência (Nº de Artigos)",
        ylabel="Família do Modelo"
    )

    # --- RQ2: Esportes ---
    df['Esporte_Limpo'] = df['Esporte'].apply(lambda x: 'Futebol' if 'Futebol' in str(x) else x)
    df['Esporte_Limpo'] = df['Esporte_Limpo'].apply(lambda x: 'Futebol' if 'Soccer' in str(x) else x)
    
    sport_abbreviations = {
        'Basquete (National Basketball Association - NBA)': 'Basquete (NBA)',
        'Tênis (ATP e WTA)': 'Tênis (ATP/WTA)',
        'Artes Marciais Mistas (MMA), especificamente UFC (Ultimate Fighting Championship)': 'MMA (UFC)',
    }
    df['Esporte_Limpo'] = df['Esporte_Limpo'].replace(sport_abbreviations)

    string_para_remover = 'Não se aplica (Foco na Escala da Indústria Esportiva - macroeconomia do esporte)'
    df_filtrado_rq2 = df[df['Esporte_Limpo'] != string_para_remover]
    
    sport_counts = df_filtrado_rq2['Esporte_Limpo'].value_counts()
    
    plot_vertical_bar(
        data=sport_counts,
        title="RQ2: Frequência de Esportes Investigados",
        filename="RQ2_Esportes_Frequencia.png",
        xlabel="Esporte",
        ylabel="Frequência (Nº de Artigos)",
        figsize=(10, 7) # Ajuste de tamanho para legibilidade dos rótulos
    )
    
    # --- RQ3: Métricas de Avaliação ---
    metric_counts = clean_and_normalize_metrics(df['Métricas de Avaliação'])
    plot_horizontal_bar(
        data=metric_counts.head(10), # Mostra o Top 10
        title="RQ3: Frequência de Métricas de Avaliação (Top 10)",
        filename="RQ3_Metricas_Frequencia.png",
        xlabel="Frequência (Nº de Artigos)",
        ylabel="Métrica"
    )

    # --- RQ4: Lacunas e Limitações (Quantitativo) ---
    limitation_counts = clean_and_normalize_limitations(df['Limitações'], df['Lacunas'])
    plot_vertical_bar(
        data=limitation_counts,
        title="RQ4: Frequência de Limitações e Lacunas Reportadas",
        filename="RQ4_Limitacoes_Frequencia.png",
        xlabel="Tema da Limitação",
        ylabel="Frequência (Nº de Artigos)",
        figsize=(12, 7) # Ajuste para legibilidade dos rótulos
    )

    # --- RQ5: Publicações por Ano ---
    year_counts = df['Ano'].value_counts().sort_index()
    plot_vertical_bar(
        data=year_counts,
        title="RQ5: Evolução das Publicações por Ano",
        filename="RQ5_Publicacoes_por_Ano.png",
        xlabel="Ano de Publicação",
        ylabel="Frequência (Nº de Artigos)"
    )

    print("\nAnálise concluída. 5 arquivos .png foram salvos no diretório.")

except FileNotFoundError:
    print("Erro: A planilha '14 - 14.csv' não foi encontrada. Verifique se o arquivo está na mesma pasta que o script.")
except Exception as e:
    print(f"Ocorreu um erro durante a análise: {e}")