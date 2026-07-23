import pandas as pd

# ==========================================
# 1. CONSTRUÇÃO DO CONJUNTO DE IMAGENS RELEVANTES
# ==========================================
# As funções de métrica abaixo (precision_at_k, recall_at_k,
# average_precision) são genéricas: não sabem nada sobre Fashion ou
# Flickr, só recebem um "ranking" (lista de caminhos de imagem, na
# ordem devolvida pelo modelo) e um "conjunto de relevantes" (quais
# caminhos de imagem são considerados corretos para aquela query).
# Essas duas funções abaixo são a ponte entre o gabarito de cada
# dataset (que têm formatos diferentes) e essa interface comum.

def relevantes_fashion(gabarito_fashion, categoria):
    """Devolve o conjunto de caminho_imagem cuja categoria bate com a
    categoria_esperada da query. Uma imagem = uma categoria no Fashion."""
    relevantes = set()
    for _, linha in gabarito_fashion.iterrows():
        if linha['categoria'] == categoria:
            relevantes.add(linha['caminho_imagem'])
    return relevantes


def relevantes_flickr(gabarito_flickr, categoria):
    """Devolve o conjunto de caminho_imagem que contém a categoria dada
    entre suas categorias (uma imagem pode ter várias, separadas por ';')."""
    relevantes = set()
    for _, linha in gabarito_flickr.iterrows():
        lista_categorias = linha['categorias'].split(';')
        if categoria in lista_categorias:
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
    # Dividimos pelo tamanho real do top_k (e não por k fixo), pois se o
    # ranking tiver menos de k itens (não deveria acontecer aqui, já que
    # os datasets têm 500 imagens, mas é uma proteção barata), dividir
    # por k penalizaria a métrica por um motivo que não é culpa do modelo.
    return relevantes_recuperados / len(top_k)


def recall_at_k(ranking, relevantes, k):
    """Das imagens relevantes que existem no dataset inteiro, quantas
    o modelo conseguiu trazer dentro do top-k? Mede o quanto o modelo
    "não deixou passar" respostas corretas."""
    if len(relevantes) == 0:
        return 0.0  # não deveria acontecer - a verificação de sanidade já garante isso

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
            precisao_nesta_posicao = relevantes_recuperados / posicao
            soma_precisoes += precisao_nesta_posicao

    if relevantes_recuperados == 0:
        return 0.0

    # Divide pelo total de relevantes que EXISTEM (não pelos que foram
    # encontrados) - isso penaliza um modelo que nunca encontra algumas
    # das imagens corretas, mesmo que as que ele encontrou estejam bem rankeadas.
    return soma_precisoes / len(relevantes)


# ==========================================
# 3. AVALIAR UMA LISTA INTEIRA DE QUERIES
# ==========================================
def avaliar_queries(queries, ranking, funcao_relevantes, gabarito, k=5):
    """
    queries: lista de dicts {"texto":..., "categoria_esperada":...}
    funcao_relevantes: relevantes_fashion ou relevantes_flickr
    gabarito: gabarito_fashion ou gabarito_flickr (DataFrame)
    """
    linhas_resultado = []

    for query in queries:
        categoria = query['categoria_esperada']
        relevantes = funcao_relevantes(gabarito, categoria)

        linhas_resultado.append({
            'texto': query['texto'],
            'categoria_esperada': categoria,
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

    print("=== Teste 1: ranking perfeito (todos relevantes primeiro) ===")
    relevantes = {"img1.jpg", "img3.jpg"}
    ranking_perfeito = ["img1.jpg", "img3.jpg", "img2.jpg", "img4.jpg", "img5.jpg"]

    p5 = precision_at_k(ranking_perfeito, relevantes, k=5)
    r5 = recall_at_k(ranking_perfeito, relevantes, k=5)
    ap = average_precision(ranking_perfeito, relevantes)
    print(f"precision@5={p5}, recall@5={r5}, AP={ap}")
    assert p5 == 2/5, "precision@5 deveria ser 2/5 (2 relevantes em 5 retornados)"
    assert r5 == 1.0, "recall@5 deveria ser 1.0 (achou os 2 relevantes que existem)"
    assert ap == 1.0, "AP deveria ser 1.0 (os 2 relevantes vieram nas 2 primeiras posições)"
    print("OK\n")

    print("=== Teste 2: nenhum relevante recuperado ===")
    ranking_ruim = ["img2.jpg", "img4.jpg", "img5.jpg"]
    p5 = precision_at_k(ranking_ruim, relevantes, k=5)
    r5 = recall_at_k(ranking_ruim, relevantes, k=5)
    ap = average_precision(ranking_ruim, relevantes)
    print(f"precision@5={p5}, recall@5={r5}, AP={ap}")
    assert p5 == 0.0 and r5 == 0.0 and ap == 0.0
    print("OK\n")

    print("=== Teste 3: ranking misturado (calculado manualmente para conferir) ===")
    # relevantes = {doc1, doc3}
    # ranking =    [doc2, doc1, doc4, doc3, doc5]
    # doc1 é relevante e está na posição 2 -> precisao ali = 1/2
    # doc3 é relevante e está na posição 4 -> precisao ali = 2/4
    # AP = (1/2 + 2/4) / 2 = 0.5
    relevantes_3 = {"doc1", "doc3"}
    ranking_3 = ["doc2", "doc1", "doc4", "doc3", "doc5"]
    ap3 = average_precision(ranking_3, relevantes_3)
    print(f"AP calculado = {ap3} | esperado manualmente = 0.5")
    assert ap3 == 0.5
    print("OK\n")

    print("=== Teste 4: com o gabarito REAL do Fashion, categoria 'Watches' ===")
    gabarito_fashion = pd.read_csv('dataset_amostra_500/fashion/fashion_gabarito.csv')
    relevantes_watches = relevantes_fashion(gabarito_fashion, "Watches")
    print(f"Quantidade real de relógios na amostra: {len(relevantes_watches)}")

    # Simula um ranking "perfeito": todos os relógios relevantes vêm
    # primeiro, seguidos de imagens irrelevantes.
    irrelevantes = list(gabarito_fashion[gabarito_fashion['categoria'] != 'Watches']['caminho_imagem'])[:10]
    ranking_simulado = list(relevantes_watches) + irrelevantes

    p5 = precision_at_k(ranking_simulado, relevantes_watches, k=5)
    ap = average_precision(ranking_simulado, relevantes_watches)
    print(f"precision@5={p5} (esperado 1.0, já que os 5 primeiros são todos relógios)")
    print(f"AP={ap} (esperado 1.0, ranking perfeito)")
    assert p5 == 1.0
    assert ap == 1.0
    print("OK\n")

    print("Todos os testes passaram.")