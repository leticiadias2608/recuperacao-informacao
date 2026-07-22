import pandas as pd
import numpy as np

from sbert_utils import codificar_textos_sbert, buscar_sbert

def main():

    df_fashion = pd.read_csv("dataset_amostra_500/fashion/fashion_processado.csv")
    df_flickr  = pd.read_csv("dataset_amostra_500/flickr/flickr_processado.csv")

    # A ordem das linhas do DataFrame define a ordem das linhas da matriz de embeddings
    sbert_fashion = codificar_textos_sbert(df_fashion["texto_busca"].tolist())
    sbert_flickr  = codificar_textos_sbert(df_flickr["texto_busca"].tolist())

    print(sbert_fashion.shape)  # esperado: (500, 384)
    print(sbert_flickr.shape)   # esperado: (500, 384)

    np.save("sbert_fashion.npy", sbert_fashion)
    np.save("sbert_flickr.npy", sbert_flickr)

    # Listas de caminhos na MESMA ordem das linhas das matrizes acima --
    # e' o que buscar_sbert usa para traduzir indice de volta em arquivo.
    caminhos_fashion = df_fashion["caminho_imagem"].tolist()
    caminhos_flickr  = df_flickr["caminho_imagem"].tolist()

    # Queries de teste so' para validar o pipeline antes de plugar o
    # conjunto de queries de verdade que o Integrante 3 vai fornecer.
    query_fashion = "red running shoes"
    query_flickr  = "a dog running on the beach"

    caminhos_rankeados_fashion, scores_fashion = buscar_sbert(
        query_fashion, sbert_fashion, caminhos_fashion, top_k=5
    )
    caminhos_rankeados_flickr, scores_flickr = buscar_sbert(
        query_flickr, sbert_flickr, caminhos_flickr, top_k=5
    )

    print(f"\nTop-5 resultados no Fashion para: '{query_fashion}'")
    for caminho, score in zip(caminhos_rankeados_fashion, scores_fashion):
        print(f"  {score:.4f}  {caminho}") # depois substituir pelas métricas

    print(f"\nTop-5 resultados no Flickr para: '{query_flickr}'")
    for caminho, score in zip(caminhos_rankeados_flickr, scores_flickr):
        print(f"  {score:.4f}  {caminho}") # depois substituir pelas métricas

if __name__ == "__main__":
    main()