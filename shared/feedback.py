import numpy as np

def rocchio_method(alfa, beta, gama, query, doc_rel, N_rel, doc_n, N_n, vocabulario):
    sum_rel = np.zeros(len(vocabulario)) # inicializa o vetor com zero
    D_r = np.array(doc_rel) # vetor de documentos relevantes
    
    sum_n = np.zeros(len(vocabulario)) # inicializa o vetor com zero
    D_n = np.array(doc_n) # vetor de documentos não relevantes
    for doc in D_r:
        sum_rel = sum_rel + doc
    
    for doc in D_n:
        sum_n = sum_n + doc
        
    q = np.array(query)

    # parcela dos relevantes: só calcula se houver documentos relevantes
    if N_rel > 0: 
        parcela_rel = (beta / N_rel) * sum_rel
    else: 
        parcela_n = np.zeros(len(vocabulario))

    # parcela dos não relevantes: só calcula se houver documentos não relevantes
    if N_n > 0:
        parcela_n = (gama / N_n) * sum_n  
    else: 
        parcela_n = np.zeros(len(vocabulario))

    modified_query = alfa*q + parcela_rel - parcela_n
    
    return modified_query

def feedback(rank, lista_docs, weighted_docs, queries_dict):

    # ve a query correspondente 
    indice = int(rank['query'])-1
    query = queries_dict[indice]
    ranking_str = rank['ranking']  # pega ranking
    
    doc_rel = []
    doc_n = []
    for doc_id_str in ranking_str:
        
        # transforma a string lida do txt em int
        doc_id = int(doc_id_str) 
        doc = lista_docs[doc_id - 1] # doc_id é 1-based
        
        # checo se é relevante ou não
        if doc['category'] == query['category']:
            doc_rel.append(weighted_docs[doc_id - 1])
        else:
            doc_n.append(weighted_docs[doc_id - 1])

    return doc_rel, doc_n