import pandas as pd
import numpy as np

from embeddings_clip import codificar_imagens_clip, buscar_clip
from sbert_utils import codificar_textos_sbert, buscar_sbert

def main():

    df_fashion = pd.read_csv("dataset_amostra_500/fashion/fashion_processado.csv")
    df_flickr  = pd.read_csv("dataset_amostra_500/flickr/flickr_processado.csv")

    # A ordem das linhas do DataFrame define a ordem das linhas da matriz de embeddings
    sbert_fashion = codificar_textos_sbert(df_fashion["texto_busca"].tolist())
    sbert_flickr  = codificar_textos_sbert(df_flickr["texto_busca"].tolist())

    np.save("sbert_fashion.npy", sbert_fashion)
    np.save("sbert_flickr.npy", sbert_flickr)

    # Listas de caminhos na MESMA ordem das linhas das matrizes acima --
    # e' o que buscar_sbert usa para traduzir indice de volta em arquivo.
    caminhos_fashion = df_fashion["caminho_imagem"].tolist()
    caminhos_flickr  = df_flickr["caminho_imagem"].tolist()
    
    # Codificar as imagens usando CLIP
    clip_img_fashion = codificar_imagens_clip(caminhos_fashion)
    clip_img_flickr = codificar_imagens_clip(caminhos_flickr)
    
    np.save("clip_img_fashion.npy", clip_img_fashion)
    np.save("clip_img_flickr.npy", clip_img_flickr)

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
        
    # CLIP
    # Busca de exemplo
    resultados_clip = buscar_clip("cachorro correndo na praia", clip_img_flickr, caminhos_flickr, top_k=5)


if __name__ == "__main__":
    main()