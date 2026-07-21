
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

MODELO_SBERT = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

""" Roda o S-BERT sobre a coluna texto_busca (tanto do Fashion quanto do Flickr) e devolve a matriz de embeddings """
def codificar_textos_sbert(lista_de_textos, batch_size=32):

    embeddings = MODELO_SBERT.encode(
        lista_de_textos,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings # np.ndarray de shape (n_textos, 384)

""" Gera o embedding da query, calcula similaridade de cosseno contra a matriz, retorna top-k caminhos de imagem ranqueados """
def buscar_sbert(query, matriz_embeddings, lista_caminhos, top_k=10):

    embedding_query = MODELO_SBERT.encode(
        [query],    # espera uma lista, mesmo que seja uma lista de um único elemento
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    similaridades = matriz_embeddings @ embedding_query

    indices_ordenados = np.argsort(-similaridades)
    top_indices = indices_ordenados[:top_k]

    caminhos_rankeados = [lista_caminhos[i] for i in top_indices]
    scores_rankeados = similaridades[top_indices]

    return caminhos_rankeados, scores_rankeados # list[str], np.ndarray