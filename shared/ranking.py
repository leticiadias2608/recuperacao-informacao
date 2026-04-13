###------------FUNÇÕES DE SIMILARIDADE------------###

import numpy as np
from shared import utils

### SIMILARIDADE COSSENO ###
def similaridade_cos(vet1, vet2):
    prod_esc = np.dot(vet1, vet2) #produto escalar entre vet1 e vet2
    
    #cálculo das normas dos vetores
    norma_vet1 = np.linalg.norm(vet1)
    norma_vet2 = np.linalg.norm(vet2)

    if norma_vet1 == 0 or norma_vet2 == 0:  # evita divisão por zero
        return 0.0

    similaridade = prod_esc / (norma_vet1 * norma_vet2)
    #print(similaridade)
    return similaridade

###------------FUNÇÕES DE RANQUEAMENTO------------###

def ranqueamento_cos(query, vetores_tf): # retorna os 30 primeiro documentos mais similares
    similaridades = [] # lista de similaridades
    indices = []

    for i, documento in enumerate(vetores_tf):
        similaridades.append(similaridade_cos(query, documento))
        indices.append(i+1) # o documento da posição i é o documento de nome i+1
    pares = list(zip(similaridades, indices)) # listas com os pares (similaridade indice_doc)
    pares_ranqueados = sorted(pares, key=lambda x: x[0], reverse=True)

    # Ranqueia documentos
    ranked_list = [doc_id for sim, doc_id in pares_ranqueados] # a lista possui os índices dos documentos

    #print(ranked_list)
    return ranked_list[:30] 


### ---------------------- ATIVIDADE 2 ------------------------ ###
import math 

###------------FUNÇÕES DE SIMILARIDADE------------###
def sim_BM1(q, document, N, ni_map):
    similaridade = 0
    for k in q:
        if k in document:
            ni = ni_map.get(k, 0)
            similaridade += math.log((N-ni+0.5)/(ni+0.5))
    return similaridade

def sim_BM11(q, document, N, ni_map, avg_doclen, K, f_vector):
    similaridade = 0
    for k in q:
        if k in document:
            f = f_vector[k]
            ni = ni_map.get(k, 0)
            parte_1 = ((K+1)*f)/(((K*len(document))/(avg_doclen))+f)
            parte_2 = math.log((N-ni+0.5)/ni+0.5)
            similaridade += parte_1 * parte_2
    return similaridade

def sim_BM15(q, document, N, ni_map, K, f_vector):
    similaridade = 0
    for k in q:
        if k in document:
            f = f_vector[k] 
            ni = ni_map.get(k, 0)
            parte_1 = (((K+1)*f)/(K+f))
            parte_2 = math.log((N-ni+0.5)/ni+0.5)
            similaridade += parte_1 * parte_2
    return similaridade

def sim_BM25(q, document, N, ni_map, avg_doclen, K, b):
    similaridade = 0
    for k in q:
        if k in document:
            ni = ni_map.get(k, 0)
            f = document.count(k)  # frequência do termo k no documento
            B_f = utils.B_frequency(K, b, f, avg_doclen, document)
            similaridade += B_f * math.log((N - ni + 0.5) / (ni + 0.5))
    return similaridade

###------------FUNÇÕES DE RANQUEAMENTO------------###

def ranqueamento_prob(q, conteudo_tokens, ni_map, N, avg_doclen, func_sim, K, b):
    """
    Calcula a similaridade da query q contra todos os documentos
    usando a função de similaridade especificada e retorna os 30 mais similares.
    func_sim: "BM1" | "BM11" | "BM15" | "BM25"
    """
    similaridades = []
 
    for i, document in enumerate(conteudo_tokens):
        if func_sim == "BM1":
            sim = sim_BM1(q, document, N, ni_map)
        elif func_sim == "BM11":    
            f_vector = {termo: document.count(termo) for termo in set(document)}  # Criar os vetores de frequencia
            sim = sim_BM11(q, document, N, ni_map, avg_doclen, K, f_vector)
        elif func_sim == "BM15":
            f_vector = {termo: document.count(termo) for termo in set(document)} # Criar os vetores de frequencia
            sim = sim_BM15(q, document, N, ni_map, K, f_vector)
        elif func_sim == "BM25":
            sim = sim_BM25(q, document, N, ni_map, avg_doclen, K, b)
        else:
            sim = 0.0
 
        similaridades.append((sim, i + 1))  # i+1 para doc_id 1-based
 
    pares_ranqueados = sorted(similaridades, key=lambda x: x[0], reverse=True)
    ranked_list = [doc_id for sim, doc_id in pares_ranqueados]
    return ranked_list[:30]