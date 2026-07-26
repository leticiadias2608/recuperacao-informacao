import os
import shutil
import pandas as pd

# ==========================================
# EXEMPLOS PARA USAR
# ==========================================
# exemplos específicos em que a mesma query, retornou 
# o resultado de um modelo bom e outro ruim. 
EXEMPLOS_A_EXTRAIR = [
    ("clip", "fashion", "especifica", "Grey striped t-shirt by Nike", 5),
    ("sbert", "fashion", "especifica", "Grey striped t-shirt by Nike", 5),

    ("clip", "flickr", "generica", "Adventure on rocky terrain", 5),
    ("sbert", "flickr", "generica", "Adventure on rocky terrain", 5)
]

PASTA_RESULTS = "results"
PASTA_SAIDA_IMAGENS = "exemplos_apresentacao"


# Lê o arquivo e retorna os caminhos das imagens do ranking
def extrair_ranking_da_query(caminho_txt, texto_query_alvo, k):
    """
    Cada linha do .txt tem o formato:  "texto da query": img1.jpg img2.jpg ...
    Procura a linha cujo texto bate exatamente com texto_query_alvo e
    devolve os k primeiros caminhos de imagem dessa linha.
    """
    with open(caminho_txt, "r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue

            # A linha é "texto": img1 img2 img3 - separa no primeiro ": " depois das aspas
            fechamento_aspas = linha.index('": ')
            texto_da_linha = linha[1:fechamento_aspas]  # remove a aspa inicial
            resto = linha[fechamento_aspas + 3:]

            if texto_da_linha == texto_query_alvo:
                caminhos = resto.split(" ")
                return caminhos[:k]

    return None  # query não encontrada nesse arquivo


# Pega o exemplo para cada configuração
def rodar_extracao():
    os.makedirs(PASTA_SAIDA_IMAGENS, exist_ok=True)

    for modelo, dataset, tipo, texto_query, k in EXEMPLOS_A_EXTRAIR:
        nome_arquivo = f"ranking_imagens_{modelo}_{dataset}_{tipo}.txt"
        caminho_txt = os.path.join(PASTA_RESULTS, nome_arquivo)

        if not os.path.exists(caminho_txt):
            print(f"AVISO: não encontrei {caminho_txt} - pulei esse exemplo")
            continue

        caminhos_top_k = extrair_ranking_da_query(caminho_txt, texto_query, k)

        if caminhos_top_k is None:
            print(f"AVISO: query '{texto_query}' não encontrada em {nome_arquivo}")
            continue

        print(f"\n=== {modelo.upper()} | {dataset} | {tipo} | \"{texto_query}\" ===")
        for posicao, caminho_imagem in enumerate(caminhos_top_k, start=1):
            print(f"  {posicao}. {caminho_imagem}")

        # Copia as imagens de verdade pra uma pasta separada, com nome
        # descritivo, pra ficar fácil de arrastar pro slide depois.
        pasta_dataset = "dataset_amostra_500/fashion/images" if dataset == "fashion" else "dataset_amostra_500/flickr/images"

        for posicao, caminho_relativo in enumerate(caminhos_top_k, start=1):
          nome_arquivo_original = os.path.basename(caminho_relativo)
          # Correção: usando pasta_dataset em vez de recriar o caminho na mão
          origem = os.path.join(pasta_dataset, caminho_relativo)
          nome_novo = (
              f"{modelo}_{dataset}_{tipo}_pos{posicao}_{nome_arquivo_original}"
          )
          destino = os.path.join(PASTA_SAIDA_IMAGENS, nome_novo)

          if os.path.exists(origem):
            shutil.copy2(origem, destino)
          else:
            print(f"  (imagem não encontrada no disco: {origem})")

    print(f"\nImagens copiadas para a pasta '{PASTA_SAIDA_IMAGENS}/' (quando encontradas no disco).")


if __name__ == "__main__":
    rodar_extracao()