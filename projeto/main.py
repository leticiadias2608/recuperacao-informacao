import pandas as pd
import numpy as np
import json
import os

from embeddings_clip import codificar_imagens_clip, buscar_clip
from sbert_utils import codificar_textos_sbert, buscar_sbert

from queries_evaluation import relevantes_fashion, relevantes_flickr, avaliar_e_salvar

def main():
    print("Carregando bases e gabaritos...")
    df_fashion = pd.read_csv("dataset_amostra_500/fashion/fashion_processado.csv")
    df_flickr  = pd.read_csv("dataset_amostra_500/flickr/flickr_processado.csv")
    
    gabarito_fashion = pd.read_csv("dataset_amostra_500/fashion/fashion_gabarito.csv")
    gabarito_flickr = pd.read_csv("dataset_amostra_500/flickr/flickr_gabarito.csv")

    caminhos_fashion = df_fashion["caminho_imagem"].tolist()
    caminhos_flickr  = df_flickr["caminho_imagem"].tolist()

    # ==========================================
    # 1. LEITURA E SEPARAÇÃO DAS QUERIES
    # ==========================================
    print("\nCarregando queries e separando listas...")
    with open("dataset_amostra_500/fashion/fashion_queries.json", "r", encoding="utf-8") as f:
        fashion_queries = json.load(f)
        
    with open("dataset_amostra_500/flickr/flickr_queries.json", "r", encoding="utf-8") as f:
        flickr_queries = json.load(f)

    # Divisão utilizando os índices (pares=genéricas, ímpares=específicas)
    generica_fashion = fashion_queries[0::2]
    especifica_fashion = fashion_queries[1::2]

    generica_flickr = flickr_queries[0::2]
    especifica_flickr = flickr_queries[1::2]

    # ==========================================
    # 2. CARREGAMENTO OU GERAÇÃO DE EMBEDDINGS (S-BERT e CLIP)
    # ==========================================

    # --- S-BERT ---
    if os.path.exists("sbert_fashion.npy"):
        print("\nCarregando embeddings S-BERT (Fashion) do disco...")
        sbert_fashion = np.load("sbert_fashion.npy")
    else:
        print("\nGerando embeddings S-BERT (Fashion)...")
        sbert_fashion = codificar_textos_sbert(df_fashion["texto_busca"].tolist())
        np.save("sbert_fashion.npy", sbert_fashion)

    if os.path.exists("sbert_flickr.npy"):
        print("Carregando embeddings S-BERT (Flickr) do disco...")
        sbert_flickr = np.load("sbert_flickr.npy")
    else:
        print("Gerando embeddings S-BERT (Flickr)...")
        sbert_flickr = codificar_textos_sbert(df_flickr["texto_busca"].tolist())
        np.save("sbert_flickr.npy", sbert_flickr)

    # --- CLIP ---
    if os.path.exists("clip_img_fashion.npy"):
        print("\nCarregando embeddings CLIP (Fashion) do disco...")
        clip_img_fashion = np.load("clip_img_fashion.npy")
    else:
        print("\nGerando embeddings CLIP (Fashion)...")
        clip_img_fashion = codificar_imagens_clip(caminhos_fashion)
        np.save("clip_img_fashion.npy", clip_img_fashion)

    if os.path.exists("clip_img_flickr.npy"):
        print("Carregando embeddings CLIP (Flickr) do disco...")
        clip_img_flickr = np.load("clip_img_flickr.npy")
    else:
        print("Gerando embeddings CLIP (Flickr)...")
        clip_img_flickr = codificar_imagens_clip(caminhos_flickr)
        np.save("clip_img_flickr.npy", clip_img_flickr)

    # ==========================================
    # 3. AVALIAÇÃO SEPARADA (MODELO x DATASET x BUSCA)
    # ==========================================
    print("\nIniciando avaliações das queries e salvando métricas...")
    k_avaliacao = 5

    # --- S-BERT ---
    avaliar_e_salvar(generica_fashion, "sbert", "fashion", "generica", sbert_fashion, caminhos_fashion, gabarito_fashion, relevantes_fashion, buscar_sbert, k=k_avaliacao)
    avaliar_e_salvar(especifica_fashion, "sbert", "fashion", "especifica", sbert_fashion, caminhos_fashion, gabarito_fashion, relevantes_fashion, buscar_sbert, k=k_avaliacao)
    
    avaliar_e_salvar(generica_flickr, "sbert", "flickr", "generica", sbert_flickr, caminhos_flickr, gabarito_flickr, relevantes_flickr, buscar_sbert, k=k_avaliacao)
    avaliar_e_salvar(especifica_flickr, "sbert", "flickr", "especifica", sbert_flickr, caminhos_flickr, gabarito_flickr, relevantes_flickr, buscar_sbert, k=k_avaliacao)

    # --- CLIP ---
    avaliar_e_salvar(generica_fashion, "clip", "fashion", "generica", clip_img_fashion, caminhos_fashion, gabarito_fashion, relevantes_fashion, buscar_clip, k=k_avaliacao)
    avaliar_e_salvar(especifica_fashion, "clip", "fashion", "especifica", clip_img_fashion, caminhos_fashion, gabarito_fashion, relevantes_fashion, buscar_clip, k=k_avaliacao)
    
    avaliar_e_salvar(generica_flickr, "clip", "flickr", "generica", clip_img_flickr, caminhos_flickr, gabarito_flickr, relevantes_flickr, buscar_clip, k=k_avaliacao)
    avaliar_e_salvar(especifica_flickr, "clip", "flickr", "especifica", clip_img_flickr, caminhos_flickr, gabarito_flickr, relevantes_flickr, buscar_clip, k=k_avaliacao)

    print("\nProcesso finalizado! Todos os arquivos CSV de resultados foram gerados.")

if __name__ == "__main__":
    main()