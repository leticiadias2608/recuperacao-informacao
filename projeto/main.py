import pandas as pd
import numpy as np
import json
import os

from embeddings_clip import codificar_imagens_clip, buscar_clip
from sbert_utils import codificar_textos_sbert, buscar_sbert

from queries_evaluation import relevantes_fashion, relevantes_flickr, avaliar_e_salvar

def main():
    print("Carregando bases e dados de gabarito...")
    pasta_amostra = "dataset_amostra_500"
    pasta_fashion_base = os.path.join(pasta_amostra, "fashion")
    pasta_flickr_base = os.path.join(pasta_amostra, "flickr")

    df_fashion = pd.read_csv(os.path.join(pasta_fashion_base, "fashion_processado.csv"))
    df_flickr  = pd.read_csv(os.path.join(pasta_flickr_base, "flickr_processado.csv"))

    dados_fashion = pd.read_csv(os.path.join(pasta_fashion_base, "fashion_dados.csv"))
    dados_flickr = pd.read_csv(os.path.join(pasta_flickr_base, "flickr_dados.csv"))

    # Deixa caminho_imagem em dados_fashion/dados_flickr no mesmo formato
    # (com o prefixo da pasta) usado em caminhos_fashion/caminhos_flickr,
    # senão relevantes_fashion/relevantes_flickr nunca batem com o ranking
    # devolvido por buscar_sbert/buscar_clip.
    dados_fashion["caminho_imagem"] = dados_fashion["caminho_imagem"].apply(
        lambda c: os.path.join(pasta_fashion_base, c)
    )
    dados_flickr["caminho_imagem"] = dados_flickr["caminho_imagem"].apply(
        lambda c: os.path.join(pasta_flickr_base, c)
    )

    caminhos_fashion = [
        os.path.join(pasta_fashion_base, caminho_relativo)
        for caminho_relativo in df_fashion["caminho_imagem"]
    ]
    caminhos_flickr = [
        os.path.join(pasta_flickr_base, caminho_relativo)
        for caminho_relativo in df_flickr["caminho_imagem"]
    ]

    # ==========================================
    # 1. LEITURA E SEPARAÇÃO DAS QUERIES
    # ==========================================
    print("\nCarregando queries e separando listas...")
    with open(os.path.join(pasta_fashion_base, "fashion_queries.json"), "r", encoding="utf-8") as f:
        fashion_queries = json.load(f)

    with open(os.path.join(pasta_flickr_base, "flickr_queries.json"), "r", encoding="utf-8") as f:
        flickr_queries = json.load(f)

    # Divisão utilizando o tipo (genéricas e específicas)
    generica_fashion = []
    especifica_fashion = []
    for q in fashion_queries:
        if q['tipo'] == 'generica':
            generica_fashion.append(q)
        else:
            especifica_fashion.append(q)
 
    generica_flickr = []
    especifica_flickr = []
    for q in flickr_queries:
        if q['tipo'] == 'generica':
            generica_flickr.append(q)
        else:
            especifica_flickr.append(q)
 
    print(f"Fashion: {len(generica_fashion)} genéricas, {len(especifica_fashion)} específicas")
    print(f"Flickr: {len(generica_flickr)} genéricas, {len(especifica_flickr)} específicas")

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
    avaliar_e_salvar(generica_fashion, "sbert", "fashion", "generica", sbert_fashion, caminhos_fashion, dados_fashion, relevantes_fashion, buscar_sbert, k=k_avaliacao)
    avaliar_e_salvar(especifica_fashion, "sbert", "fashion", "especifica", sbert_fashion, caminhos_fashion, dados_fashion, relevantes_fashion, buscar_sbert, k=k_avaliacao)
    
    avaliar_e_salvar(generica_flickr, "sbert", "flickr", "generica", sbert_flickr, caminhos_flickr, dados_flickr, relevantes_flickr, buscar_sbert, k=k_avaliacao)
    avaliar_e_salvar(especifica_flickr, "sbert", "flickr", "especifica", sbert_flickr, caminhos_flickr, dados_flickr, relevantes_flickr, buscar_sbert, k=k_avaliacao)

    # --- CLIP ---
    avaliar_e_salvar(generica_fashion, "clip", "fashion", "generica", clip_img_fashion, caminhos_fashion, dados_fashion, relevantes_fashion, buscar_clip, k=k_avaliacao)
    avaliar_e_salvar(especifica_fashion, "clip", "fashion", "especifica", clip_img_fashion, caminhos_fashion, dados_fashion, relevantes_fashion, buscar_clip, k=k_avaliacao)
    
    avaliar_e_salvar(generica_flickr, "clip", "flickr", "generica", clip_img_flickr, caminhos_flickr, dados_flickr, relevantes_flickr, buscar_clip, k=k_avaliacao)
    avaliar_e_salvar(especifica_flickr, "clip", "flickr", "especifica", clip_img_flickr, caminhos_flickr, dados_flickr, relevantes_flickr, buscar_clip, k=k_avaliacao)

    print("\nProcesso finalizado! Todos os arquivos CSV de resultados foram gerados.")

if __name__ == "__main__":
    main()