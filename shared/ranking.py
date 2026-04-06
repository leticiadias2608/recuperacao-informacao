###------------FUNÇÕES DE SIMILARIDADE------------###

import numpy as np

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

def ranqueamento(query, vetores_tf, documentos): # retorna os 30 primeiro documentos mais similares
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
