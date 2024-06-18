import pandas as pd
from pylatexenc.latexwalker import LatexWalker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns

# Tokenize LaTeX function
def tokenize_latex(latex_expr):
    if pd.isna(latex_expr) or latex_expr.strip() == '':
        return ''
    walker = LatexWalker(latex_expr)
    nodelist, _, _ = walker.get_latex_nodes()
    tokens = [node.__str__() for node in nodelist]
    return ' '.join(tokens)

def handle_text_and_tex(df):
    df['tokenized_latex'] = df['latex'].apply(tokenize_latex)
    vectorizer_plaintext = TfidfVectorizer(max_features=1000, stop_words=None)
    X_plaintext = vectorizer_plaintext.fit_transform(df['text'])
    vectorizer_latex = TfidfVectorizer(max_features=1000, stop_words=None)
    X_latex = vectorizer_latex.fit_transform(df['tokenized_latex'])
    X_table = df[['table']]
    X_viz = df[['viz']]
    X_combined = hstack([X_plaintext, X_latex, X_table, X_viz])
    X_combined_dense = X_combined.toarray()
    return X_combined_dense

def PCA(df, num_comp=2):
    pca = PCA(n_components=num_comp)
    df_pca = pca.fit_transform(df)
    return df_pca

def execute_kmeans(df, matrix, num_clusters=3):
    kmeans = KMeans(n_clusters=num_clusters)
    clusters = kmeans.fit_predict(matrix)
    df['cluster'] = clusters
    return df

def save_clusters(df, clust_dict):
    clust_counter = 0
    for clust in clust_dict:
        clust_dict[clust] = df[df['cluster'] == clust_counter]['id']
        clust_counter += 1
    
    return clust_dict 
    

# Visualization
def basic_viz(matrix, df):
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=matrix[:, 0], y=matrix[:, 1], hue=df['cluster'], palette='viridis')
    plt.title('Clustering of SAT Questions')
    plt.show()
