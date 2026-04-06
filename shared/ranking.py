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

def ranqueamento(query, vetores_tf, documentos):
    similaridades = [] # lista de similaridades

    for i, documento in enumerate(vetores_tf):
        similaridades.append(similaridade_cos(query, documento))
    pares = list(zip(similaridades, documentos)) # listas com os pares
    pares_ranqueados = sorted(pares, key=lambda x: x[0], reverse=True)

    # Ranqueia documentos
    ranked_list = [doc for sim, doc in pares_ranqueados] 

    #print(ranked_list)
    return ranked_list
