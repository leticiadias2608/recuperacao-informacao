import pandas as pd
import matplotlib.pyplot as plt
import os

PASTA_RESULTADOS = "results"
PASTA_SAIDA = "graficos"
os.makedirs(PASTA_SAIDA, exist_ok=True)

MODELOS = ["clip", "sbert"]
DATASETS = ["fashion", "flickr"]
TIPOS = ["generica", "especifica"]

# ==========================================
# 1. TABELA FINAL DE RESUMO DOS RESULTADOS
# ==========================================
# Leitura dos arquivos de resultado e calcula a média das 3 métricas
# para cada combinação (modelo x dataset x tipo de busca)
linhas_resumo = []
for modelo in MODELOS:
    for dataset in DATASETS:
        for tipo in TIPOS:
            nome_arquivo = f"resultados_{modelo}_{dataset}_{tipo}.csv"
            caminho = os.path.join(PASTA_RESULTADOS, nome_arquivo)
            df = pd.read_csv(caminho)

            nome_modelo = "CLIP" if modelo == "clip" else "S-BERT"
            nome_tipo = "Genérica" if tipo == "generica" else "Específica"

            linhas_resumo.append({
                "modelo": nome_modelo,
                "dataset": dataset.capitalize(),
                "tipo_busca": nome_tipo,
                "precision_at_5": df["precision_at_k"].mean(),
                "recall_at_5": df["recall_at_k"].mean(),
                "MAP": df["average_precision"].mean(),
                "n_queries": len(df),
            })

tabela_resumo = pd.DataFrame(linhas_resumo)
tabela_resumo["precision_at_5"] = tabela_resumo["precision_at_5"].round(4)
tabela_resumo["recall_at_5"] = tabela_resumo["recall_at_5"].round(4)
tabela_resumo["MAP"] = tabela_resumo["MAP"].round(4)

caminho_tabela = os.path.join(PASTA_SAIDA, "tabela_resumo_metricas.csv")
tabela_resumo.to_csv(caminho_tabela, index=False)

print("=== Tabela resumo (também salva em graficos/tabela_resumo_metricas.csv) ===")
print(tabela_resumo.to_string(index=False))

# ==========================================
# 2. GRÁFICO MAP CLIP X S-BERT (genérica -> específica):
# Mostra como o valor do MAP se comporta com a mudança de busca genérica para busca 
# específica, por modelo, em cada dataset.
# ==========================================
fig, eixos = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
ordem_tipo = ["Genérica", "Específica"]

for i, dataset in enumerate(["Fashion", "Flickr"]):
    eixo = eixos[i]
    sub_dataset = tabela_resumo[tabela_resumo["dataset"] == dataset]

    for modelo in ["CLIP", "S-BERT"]:
        sub_modelo = sub_dataset[sub_dataset["modelo"] == modelo]
        sub_modelo = sub_modelo.set_index("tipo_busca").loc[ordem_tipo].reset_index()
        eixo.plot(sub_modelo["tipo_busca"], sub_modelo["MAP"], marker='o', linewidth=2.5,
                  markersize=9, label=modelo)

    eixo.set_title(dataset, fontsize=13)
    eixo.set_ylabel("MAP" if i == 0 else "")
    eixo.set_ylim(0, 1.05)
    eixo.grid(alpha=0.3)
    eixo.legend()

fig.suptitle("MAP ao mudar de query genérica para específica", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, "grafico1_interacao_map.png"), dpi=150)
plt.close()
print("\nSalvo: graficos/grafico1_interacao_map.png")

# ==========================================
# 3. GRÁFICO VISÃO GERAL
# Mostra os valores do MAP das 8 combinações em um único gráfico
# ==========================================
tabela_resumo["grupo"] = tabela_resumo["dataset"] + " · " + tabela_resumo["tipo_busca"]
grupos = list(tabela_resumo["grupo"].unique())
largura_barra = 0.35

fig, eixo = plt.subplots(figsize=(10, 5))

for indice_modelo, modelo in enumerate(["CLIP", "S-BERT"]):
    valores = []
    for grupo in grupos:
        linha = tabela_resumo[(tabela_resumo["grupo"] == grupo) & (tabela_resumo["modelo"] == modelo)]
        valores.append(linha["MAP"].values[0])

    posicoes = [indice_grupo + indice_modelo * largura_barra for indice_grupo in range(len(grupos))]
    eixo.bar(posicoes, valores, width=largura_barra, label=modelo)

posicoes_rotulo = [indice_grupo + largura_barra / 2 for indice_grupo in range(len(grupos))]
eixo.set_xticks(posicoes_rotulo)
eixo.set_xticklabels(grupos, rotation=10)
eixo.set_ylabel("MAP")
eixo.set_ylim(0, 1.05)
eixo.set_title("MAP por modelo, dataset e tipo de busca")
eixo.legend()
eixo.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PASTA_SAIDA, "grafico2_map_todas_combinacoes.png"), dpi=150)
plt.close()
print("Salvo: graficos/grafico2_map_todas_combinacoes.png")

print("\nPronto! Os 3 arquivos estão na pasta 'graficos/'.")