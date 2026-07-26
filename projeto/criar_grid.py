import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ==========================================
# CONFIGURAÇÃO
# ==========================================
PASTA_RESULTS = "results"
PASTA_SAIDA = "graficos"
K = 5

REGRAS_CATEGORIA_FLICKR = {
    'dog': r'\bdog[s]?\b', 'snow': r'\bsnow(?:y|ing)?\b',
    'water_beach': r'\b(beach(?:es)?|ocean[s]?|lake[s]?|pool[s]?|swim(?:s|ming)?|water)\b',
    'child': r'\b(child|children|kids?|boys?|girls?|bab(?:y|ies))\b',
    'man': r'\b(man|men)\b', 'woman': r'\b(woman|women)\b',
    'bicycle': r'\b(?:bik(?:e|es|er|ers|ing)|bicycle[s]?|bicyclist[s]?|cyclist[s]?)\b',
    'running': r'\brun(?:s|ning)?\b',
    'mountain_rock': r'\b(mountains?|rocks?|cliffs?)\b',
    'ball_sport': r'\b(balls?|soccer|football|basketball)\b',
    'group_people': r'\b(groups?|people|persons?|crowds?)\b',
    'grass_field': r'\b(grass|fields?|parks?)\b',
    'jumping': r'\bjump(?:s|ing)?\b', 'red_clothing': r'\bred\b',
    'playing': r'\bplay(?:s|ing|ed)?\b',
}


# ==========================================
# 1. LER O RANKING DE UM MODELO PARA UMA QUERY
# ==========================================
def extrair_ranking(caminho_txt, texto_query_alvo, k):
    with open(caminho_txt, "r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            fechamento = linha.index('": ')
            if linha[1:fechamento] == texto_query_alvo:
                return linha[fechamento + 3:].split(" ")[:k]
    return []


# ==========================================
# 2. RELEVÂNCIA - UMA FUNÇÃO PARA CADA DATASET
# (mesma lógica de relevantes_fashion/relevantes_flickr do queries_evaluation.py)
# ==========================================
def relevantes_flickr(categoria_esperada):
    df = pd.read_csv("dataset_amostra_500/flickr/flickr_processado.csv")
    padrao = REGRAS_CATEGORIA_FLICKR[categoria_esperada]

    relevantes = set()
    for _, linha in df.iterrows():
        legenda = str(linha["texto_busca"]).lower()
        if re.search(padrao, legenda):
            relevantes.add(os.path.basename(linha["caminho_imagem"]))
    return relevantes


def relevantes_fashion(atributos_esperados):
    dados = pd.read_csv("dataset_amostra_500/fashion/fashion_dados.csv")

    relevantes = set()
    for _, linha in dados.iterrows():
        texto_produto = f"{linha['productDisplayName']} {linha['baseColour']} {linha['gender']}".lower()

        bate_todos_atributos = True
        for atributo in atributos_esperados:
            if atributo.lower() not in texto_produto:
                bate_todos_atributos = False
                break

        if bate_todos_atributos:
            relevantes.add(os.path.basename(linha["caminho_imagem"]))
    return relevantes


# ==========================================
# 3. MONTAR UM GRID
# ==========================================
def montar_grid(query_texto, dataset, pasta_imagens, relevantes, modelos, tipo_busca, nome_arquivo_saida):
    fig, eixos = plt.subplots(len(modelos), K, figsize=(3 * K, 3.4 * len(modelos)))
    if len(modelos) == 1:
        eixos = [eixos]

    for linha_idx, modelo in enumerate(modelos):
        nome_arquivo_txt = f"ranking_imagens_{modelo}_{dataset}_{tipo_busca}.txt"
        ranking = extrair_ranking(os.path.join(PASTA_RESULTS, nome_arquivo_txt), query_texto, K)

        for coluna_idx in range(K):
            eixo = eixos[linha_idx][coluna_idx]
            if coluna_idx < len(ranking):
                nome_arquivo = ranking[coluna_idx]
                caminho_completo = os.path.join(pasta_imagens, nome_arquivo)

                if os.path.exists(caminho_completo):
                    imagem = Image.open(caminho_completo)
                    eixo.imshow(imagem)

                e_relevante = nome_arquivo in relevantes
                cor_borda = "green" if e_relevante else "red"
                for spine in eixo.spines.values():
                    spine.set_edgecolor(cor_borda)
                    spine.set_linewidth(4)

            eixo.set_xticks([])
            eixo.set_yticks([])

        eixos[linha_idx][0].set_ylabel(modelo.upper(), fontsize=13, fontweight="bold")

    fig.suptitle(f'"{query_texto}"', fontsize=14)
    plt.tight_layout()

    os.makedirs(PASTA_SAIDA, exist_ok=True)
    caminho_saida = os.path.join(PASTA_SAIDA, nome_arquivo_saida)
    plt.savefig(caminho_saida, dpi=150)
    plt.close()
    print(f"Salvo: {caminho_saida}")


if __name__ == "__main__":
    # --- Grid 1: Flickr, genérica ---
    rel_flickr = relevantes_flickr("mountain_rock")
    montar_grid(
        query_texto="Adventure on rocky terrain",
        dataset="flickr",
        pasta_imagens="dataset_amostra_500/flickr/images",
        relevantes=rel_flickr,
        modelos=["clip", "sbert"],
        tipo_busca="generica",
        nome_arquivo_saida="grid_adventure_on_rocky_terrain.png",
    )

    # --- Grid 2: Fashion, específica ---
    rel_fashion = relevantes_fashion(["Nike", "Grey", "Striped"])
    montar_grid(
        query_texto="Grey striped t-shirt by Nike",
        dataset="fashion",
        pasta_imagens="dataset_amostra_500/fashion/images",
        relevantes=rel_fashion,
        modelos=["clip", "sbert"],
        tipo_busca="especifica",
        nome_arquivo_saida="grid_grey_striped_tshirt_nike.png",
    )