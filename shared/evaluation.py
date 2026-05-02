import math

def precision_calc(queries_dict, ranked_lists, lista_docs, n_docs):
    lista_precisao = []
    for rank in ranked_lists:
        # ve a query correspondente 
        indice = int(rank['query'])-1
        query = queries_dict[indice]
        ranking_str = rank['ranking']  # pega ranking
        
        # pega apenas os top 'n_docs'
        top_ndocs = ranking_str[:n_docs]
        
        relevantes = 0
        
        # pega até n_docs documentos
        for doc_id_str in top_ndocs:
            
            # transforma a string lida do txt em int
            doc_id = int(doc_id_str) 
            doc = lista_docs[doc_id - 1] # doc_id é 1-based
            
            # checo se é relevante ou não
            if doc['category'] == query['category']:
                relevantes += 1
                
        precisao = relevantes / n_docs # precisao da query
        lista_precisao.append(precisao)

    media = sum(lista_precisao) / len(lista_precisao) 
    return media

def recall_calc(queries_dict, ranked_lists, lista_docs, n_docs):
    recalls = []

    for rank in ranked_lists:
        relevantes = 0
        indice = int(rank['query'])-1
        query = queries_dict[indice]
        ranking_str = rank['ranking']  # pega ranking
        for doc in lista_docs:
            # contar quantos documentos da lista_docs pertencem a categoria da query
            if doc['category'] == query['category']:
                relevantes += 1
        if relevantes == 0:
            recalls.append(0.0)
            continue

        top_ndocs = ranking_str[:n_docs]
        retrieved_relevant = 0
        
        # contar na lista de ranking quantos pertencem a essa mesma categoria
        for doc_id_str in top_ndocs:
            # transforma a string lida do txt em int
            doc_id = int(doc_id_str) 
            doc = lista_docs[doc_id - 1]
            if doc['category'] == query['category']:
                retrieved_relevant += 1

        recall = retrieved_relevant / relevantes
        recalls.append(recall)

    media = sum(recalls) / len(recalls)
    return media

def map_calc(queries_dict, ranked_lists, lista_docs):
    aps = []

    for rank in ranked_lists:
        indice = int(rank['query'])-1
        query = queries_dict[indice]
        ranking_str = rank['ranking']  # pega ranking
        
        relevantes = 0
        for doc in lista_docs:
            # contar quantos documentos da lista_docs pertencem a categoria da query
            if doc['category'] == query['category']:
                relevantes += 1
        if relevantes == 0:
            aps.append(0.0)
            continue
        relevant_found = 0
        precision_sum = 0.0

        # percorre a ranked list inteira da query
        for rank_pos, doc_id_str in enumerate(ranking_str, start=1):
            doc_id = int(doc_id_str)
            doc = lista_docs[doc_id - 1]
            if doc['category'] == query['category']:
                relevant_found += 1
                # calcula a precisão exatamente neste ponto do rank
                precision_sum += relevant_found / rank_pos

        ap = precision_sum / relevantes
        aps.append(ap)

    media = sum(aps) / len(aps)
    return media

def ndcg_calc(queries_dict, ranked_lists, lista_docs, n_docs):
    ndcgs = []

    for rank in ranked_lists:
        
        indice = int(rank['query']) - 1
        query = queries_dict[indice]
        ranking_str = rank['ranking']
        # DCG real
        dcg = 0.0
        for rank_pos, doc_id_str in enumerate(ranking_str[:n_docs], start=1):
            doc_id = int(doc_id_str)
            doc = lista_docs[doc_id - 1]
            
            if doc['category'] == query['category']:
                dcg += 1 / math.log2(rank_pos + 1)

        # DCG ideal - quantos relevantes existem até o rank k?
        total_relevant = 0
        for doc in lista_docs:
            # contar quantos documentos da lista_docs pertencem a categoria da query
            if doc['category'] == query['category']:
                total_relevant += 1
                
        ideal_relevant_at_k = min(total_relevant, n_docs)
        idcg = 0.0
        for rank_pos in range(1, ideal_relevant_at_k + 1):
            idcg += 1 / math.log2(rank_pos + 1)

        if idcg == 0:
            ndcgs.append(0.0)
            continue

        ndcgs.append(dcg / idcg)

    media = sum(ndcgs) / len(ndcgs)
    return media


def evaluate(queries_dict, ranked_lists, lista_documentos):
    """
    Calcula todas as métricas de avaliação e retorna um dicionário com os resultados.
    """
    return {
        "Precision@10": precision_calc(queries_dict, ranked_lists, lista_documentos, 10),
        "Precision@20": precision_calc(queries_dict, ranked_lists, lista_documentos, 20),
        "Precision@30": precision_calc(queries_dict, ranked_lists, lista_documentos, 30),
        "Recall@30":    recall_calc(queries_dict, ranked_lists, lista_documentos, 30),
        "MAP":          map_calc(queries_dict, ranked_lists, lista_documentos),
        "NDCG@10":      ndcg_calc(queries_dict, ranked_lists, lista_documentos, 10),
    }