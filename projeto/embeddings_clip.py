import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Carrega modelo e processor uma única vez (reaproveitar entre datasets)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"

_clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
_clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
_clip_model.eval()

def codificar_imagens_clip(lista_de_caminhos, batch_size=32):
    """
    Passa as imagens reais pelo codificador visual do CLIP e retorna
    a matriz de embeddings (n_imagens, dim_embedding), já normalizada.
    """
    todos_embeddings = []

    for i in range(0, len(lista_de_caminhos), batch_size):
        batch_caminhos = lista_de_caminhos[i:i + batch_size]
        imagens = [Image.open(caminho).convert("RGB") for caminho in batch_caminhos]

        inputs = _clip_processor(images=imagens, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            saida = _clip_model.get_image_features(**inputs, return_dict=True)
            embeddings = saida.pooler_output
            # normaliza para poder usar similaridade de cosseno via produto interno
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        todos_embeddings.append(embeddings.cpu().numpy())

    matriz_embeddings = np.vstack(todos_embeddings)
    return matriz_embeddings


def buscar_clip(query, matriz_embeddings, lista_caminhos, top_k=None):
    """
    Codifica a query em texto pelo encoder de texto do CLIP, calcula
    similaridade de cosseno contra a matriz de embeddings de imagem
    e retorna os top-k caminhos ranqueados.
    """
    if top_k is None:
        top_k = len(lista_caminhos)

    inputs = _clip_processor(text=[query], return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        saida = _clip_model.get_text_features(**inputs, return_dict=True)
        query_embedding = saida.pooler_output
        query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)

    query_embedding = query_embedding.cpu().numpy()  # shape (1, dim)

    # como ambos já estão normalizados, produto interno = similaridade de cosseno
    similaridades = matriz_embeddings @ query_embedding.T  # shape (n_imagens, 1)
    similaridades = similaridades.flatten()

    indices_ordenados = np.argsort(-similaridades)[:top_k]

    caminhos_rankeados = [lista_caminhos[i] for i in indices_ordenados]
    scores_rankeados = similaridades[indices_ordenados]
 
    return caminhos_rankeados, scores_rankeados

def main():
    pasta_amostra = 'dataset_amostra_500'
    pasta_fashion_base = os.path.join(pasta_amostra, 'fashion')
    pasta_flickr_base = os.path.join(pasta_amostra, 'flickr')

    df_fashion_final = pd.read_csv(os.path.join(pasta_fashion_base, 'fashion_processado.csv'))
    caminhos_fashion = [
        os.path.join(pasta_fashion_base, caminho_relativo)
        for caminho_relativo in df_fashion_final['caminho_imagem']
    ]

    df_flickr_final = pd.read_csv(os.path.join(pasta_flickr_base, 'flickr_processado.csv'))
    caminhos_flickr = [
        os.path.join(pasta_flickr_base, caminho_relativo)
        for caminho_relativo in df_flickr_final['caminho_imagem']
    ]

    # Fashion
    clip_img_fashion = codificar_imagens_clip(caminhos_fashion)
    np.save("clip_img_fashion.npy", clip_img_fashion)

    # Flickr8k
    clip_img_flickr = codificar_imagens_clip(caminhos_flickr)
    np.save("clip_img_flickr.npy", clip_img_flickr)

    # Busca de exemplo
    resultados = buscar_clip("cachorro correndo na praia", clip_img_flickr, caminhos_flickr, top_k=5)
    print(resultados)


if __name__ == "__main__":
    main()