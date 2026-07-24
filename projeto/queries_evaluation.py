import os

import pandas as pd

# ==========================================
# 1. CONSTRUÇÃO DO CONJUNTO DE IMAGENS RELEVANTES
# ==========================================
# As funções de métrica abaixo (precision_at_k, recall_at_k,
# average_precision) são genéricas: não sabem nada sobre Fashion ou
# Flickr, só recebem um "ranking" (lista de caminhos de imagem, na
# ordem devolvida pelo modelo) e um "conjunto de relevantes" (quais
# caminhos de imagem são considerados corretos para aquela query).
# Essas duas funções abaixo são a ponte entre os dados de cada
# dataset (que têm formatos diferentes) e essa interface comum.

def relevantes_fashion(dados_fashion, query):
    """Devolve o conjunto de caminho_imagem cujos dados batem com a categoria_esperada
    e atributos_esperados da query. Uma imagem = uma categoria no Fashion."""
    """dados_fashion: DataFrame com caminho_imagem, articleType, productDisplayName, 
    baseColour, gender (vem de fashion_dados.csv)."""
    relevantes = set()
 
    if query['tipo'] == 'generica':
        for _, linha in dados_fashion.iterrows():
            if linha['articleType'] == query['categoria_esperada']:
                relevantes.add(linha['caminho_imagem'])
    else:
        for _, linha in dados_fashion.iterrows():
            texto_produto = f"{linha['productDisplayName']} {linha['baseColour']} {linha['gender']}".lower()
 
            bate_todos_atributos = True
            for atributo in query['atributos_esperados']:
                if atributo.lower() not in texto_produto:
                    bate_todos_atributos = False
                    break
 
            if bate_todos_atributos:
                relevantes.add(linha['caminho_imagem'])
 
    return relevantes


def relevantes_flickr(dados_flickr, query):
    """Devolve o conjunto de caminho_imagem  cujos dados batem com a categoria_esperada
    e atributos_esperados da query. Uma imagem pode ter várias categorias, separadas por ';'."""
    """dados_flickr: DataFrame com caminho_imagem, legenda, categorias (vem de flickr_dados.csv)."""
    relevantes = set()
 
    if query['tipo'] == 'generica':
        for _, linha in dados_flickr.iterrows():
            val_categoria = linha['categorias']
            if pd.isna(val_categoria) or not val_categoria:
                continue
            lista_categorias = str(val_categoria).split(';')
            if query['categoria_esperada'] in lista_categorias:
                relevantes.add(linha['caminho_imagem'])
    else:
        for _, linha in dados_flickr.iterrows():
            legenda = str(linha['legenda']).lower()
 
            bate_todos_atributos = True
            for atributo in query['atributos_esperados']:
                if atributo.lower() not in legenda:
                    bate_todos_atributos = False
                    break
 
            if bate_todos_atributos:
                relevantes.add(linha['caminho_imagem'])
 
    return relevantes


# ==========================================
# 2. MÉTRICAS DE AVALIAÇÃO
# ==========================================
# Convenção usada nas três funções:
#   ranking    -> lista de caminho_imagem, JÁ ORDENADA da mais similar
#                 para a menos similar (é o que buscar_sbert/buscar_clip
#                 devolvem)
#   relevantes -> um set() de caminho_imagem que são a resposta correta
#                 para aquela query (vem de relevantes_fashion/flickr)

def precision_at_k(ranking, relevantes, k):
    """Das top-k imagens retornadas, qual fração é realmente relevante?
    Mede o quanto o resultado retornado é "limpo" (sem lixo)."""
    top_k = ranking[:k]

    relevantes_recuperados = 0
    for imagem in top_k:
        if imagem in relevantes:
            relevantes_recuperados += 1

    if len(top_k) == 0:
        return 0.0
    
    return relevantes_recuperados / len(top_k)


def recall_at_k(ranking, relevantes, k):
    """Das imagens relevantes que existem no dataset inteiro, quantas
    o modelo conseguiu trazer dentro do top-k? Mede o quanto o modelo
    "não deixou passar" respostas corretas."""
    if len(relevantes) == 0:
        return 0.0  

    top_k = ranking[:k]

    relevantes_recuperados = 0
    for imagem in top_k:
        if imagem in relevantes:
            relevantes_recuperados += 1

    return relevantes_recuperados / len(relevantes)


def average_precision(ranking, relevantes):
    """Precision média calculada em CADA posição onde uma imagem
    relevante aparece no ranking (não só no top-k fixo). Prêmia modelos
    que colocam as imagens relevantes o mais cedo possível no ranking."""
    if len(relevantes) == 0:
        return 0.0

    relevantes_recuperados = 0
    soma_precisoes = 0.0

    for posicao, imagem in enumerate(ranking, start=1):
        if imagem in relevantes:
            relevantes_recuperados += 1
            soma_precisoes += relevantes_recuperados / posicao

    if relevantes_recuperados == 0:
        return 0.0

    # Divide pelo total de relevantes que EXISTEM (não pelos que foram
    # encontrados) - isso penaliza um modelo que nunca encontra algumas
    # das imagens corretas, mesmo que as que ele encontrou estejam bem rankeadas.
    return soma_precisoes / len(relevantes)


# ==========================================
# 3. AVALIAR UMA LISTA INTEIRA DE QUERIES
# ==========================================
def avaliar_queries(queries, ranking, funcao_relevantes, dados, k=5):
    """
    queries: lista de dicts {"texto":..., "categoria_esperada":...}
    funcao_relevantes: relevantes_fashion ou relevantes_flickr
    dados: dados_fashion ou dados_flickr (DataFrame)
    """
    linhas_resultado = []

    for query in queries:
        relevantes = funcao_relevantes(dados, query)

        linhas_resultado.append({
            'texto': query['texto'],
            'tipo': query['tipo'],
            'categoria_esperada': query['categoria_esperada'],
            'precision_at_k': precision_at_k(ranking, relevantes, k),
            'recall_at_k': recall_at_k(ranking, relevantes, k),
            'average_precision': average_precision(ranking, relevantes),
        })

    df_resultado = pd.DataFrame(linhas_resultado)
    return df_resultado


# ==========================================
# 4. TESTES COM DADOS FICTÍCIOS
# (rodar este arquivo direto testa as funções ANTES de plugar os
# rankings reais do S-BERT/CLIP - pega erro de lógica cedo)
# ==========================================
if __name__ == "__main__":

    print("=== Teste 1: ranking perfeito ===")
    relevantes = {"img1.jpg", "img3.jpg"}
    ranking_perfeito = ["img1.jpg", "img3.jpg", "img2.jpg", "img4.jpg", "img5.jpg"]
    assert precision_at_k(ranking_perfeito, relevantes, 5) == 2/5
    assert recall_at_k(ranking_perfeito, relevantes, 5) == 1.0
    assert average_precision(ranking_perfeito, relevantes) == 1.0
    print("OK\n")
 
    print("=== Teste 2: nenhum relevante recuperado ===")
    ranking_ruim = ["img2.jpg", "img4.jpg", "img5.jpg"]
    assert precision_at_k(ranking_ruim, relevantes, 5) == 0.0
    assert recall_at_k(ranking_ruim, relevantes, 5) == 0.0
    assert average_precision(ranking_ruim, relevantes) == 0.0
    print("OK\n")
 
    print("=== Teste 3: AP calculado na mão ===")
    relevantes_3 = {"doc1", "doc3"}
    ranking_3 = ["doc2", "doc1", "doc4", "doc3", "doc5"]
    assert average_precision(ranking_3, relevantes_3) == 0.5
    print("OK\n")
 
    print("=== Teste 4: relevantes_fashion GENÉRICA com dados reais (Watches) ===")
    dados_fashion = pd.read_csv('dataset_amostra_500/fashion/fashion_dados.csv')
    query_generica = {"tipo": "generica", "categoria_esperada": "Watches", "atributos_esperados": []}
    rel = relevantes_fashion(dados_fashion, query_generica)
    print(f"Relógios encontrados: {len(rel)} (esperado 27)")
    assert len(rel) == 27
    print("OK\n")
 
    print("=== Teste 5: relevantes_fashion ESPECÍFICA - falso positivo corrigido? ===")
    query_especifica = {"tipo": "especifica", "categoria_esperada": "Tshirts",
                         "atributos_esperados": ["Nike", "Grey", "Striped"]}
    rel_especifica = relevantes_fashion(dados_fashion, query_especifica)
    print(f"Produtos Nike+Grey+Striped: {len(rel_especifica)} (esperado 1)")
    assert len(rel_especifica) == 1
    print("OK\n")
 
    print("Todos os testes passaram.")

# Funções auxiliares para salvar as ranked lists em .txt
def salvar_ranking_imagens_txt(resultados_busca, caminho_arquivo):

    """
    resultados_busca: lista de dicts {"texto": str, "caminhos_rankeados": list[str]}
    Salva um .txt no formato:
    "texto da query": img1 img2 img3
    """
    with open(caminho_arquivo, "w", encoding="utf-8") as f:
        for r in resultados_busca:
            imagens_str = " ".join(os.path.basename(c) for c in r["caminhos_rankeados"])
            f.write(f'"{r["texto"]}": {imagens_str}\n')

def salvar_ranking_scores_txt(resultados_busca, caminho_arquivo):

    """
    resultados_busca: lista de dicts {"texto": str, "scores_rankeados": array-like[float]}
    Salva um .txt no formato:
    "texto da query": score1 score2 score3
    """

    with open(caminho_arquivo, "w", encoding="utf-8") as f:
        for r in resultados_busca:
            scores_str = " ".join(f"{float(s):.4f}" for s in r["scores_rankeados"])
            f.write(f'"{r["texto"]}": {scores_str}\n')


""" Função principal para gerar os ranks e calcular métricas """
def avaliar_e_salvar(lista_queries, modelo_nome, dataset_nome, tipo_busca, matriz_embeddings, caminhos, dados, funcao_relevantes, func_busca, k=5):
    """
    Roda o ranqueamento para cada query individualmente, calcula as métricas 
    e salva o resultado em um CSV separado.
    """
    os.makedirs("results", exist_ok=True)
    
    resultados_metricas = []
    resultados_busca = []
    
    for q in lista_queries:
        texto_query = q["texto"]
        categoria = q["categoria_esperada"]

        # 1. Obter ranqueamento do modelo para a query atual
        caminhos_rankeados, scores = func_busca(texto_query, matriz_embeddings, caminhos, top_k=None)

        # 2. Obter conjunto de imagens relevantes do dados
        relevantes = funcao_relevantes(dados, q)

        # 3. Calcular as métricas
        p_at_k = round(precision_at_k(caminhos_rankeados, relevantes, k), 4)
        r_at_k = round(recall_at_k(caminhos_rankeados, relevantes, k), 4)
        ap = round(average_precision(caminhos_rankeados, relevantes), 4)

        resultados_metricas.append({
            "texto_query": texto_query,
            "categoria_esperada": categoria,
            "precision_at_k": p_at_k,
            "recall_at_k": r_at_k,
            "average_precision": ap
        })
        
        resultados_busca.append({
            "texto": texto_query,
            "caminhos_rankeados": caminhos_rankeados,
            "scores_rankeados": scores
        })

    # Salvar métricas consolidadas em CSV
    df_resultados = pd.DataFrame(resultados_metricas)
    nome_csv = f"resultados_{modelo_nome}_{dataset_nome}_{tipo_busca}.csv"
    df_resultados.to_csv(os.path.join("results", nome_csv), index=False)

    # Salvar as ranked lists (imagens e scores) em .txt
    nome_txt_imagens = f"ranking_imagens_{modelo_nome}_{dataset_nome}_{tipo_busca}.txt"
    nome_txt_scores = f"ranking_scores_{modelo_nome}_{dataset_nome}_{tipo_busca}.txt"
    salvar_ranking_imagens_txt(resultados_busca, os.path.join("results", nome_txt_imagens))
    salvar_ranking_scores_txt(resultados_busca, os.path.join("results", nome_txt_scores))
    print(f"Salvo: {nome_csv}, {nome_txt_imagens}, {nome_txt_scores} ({len(lista_queries)} queries)")