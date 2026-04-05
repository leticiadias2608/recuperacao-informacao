###------------FUNÇÕES DE GERAÇÃO VETORES DE PESOS------------###
import math

### VETORIZAÇÃO ###
def vetorizacao_tf(conteudo, vocabulario):
    vetor_tf = [0] * len(vocabulario)
    for termo_conteudo in conteudo:
        for i, termo in enumerate(vocabulario):
            if termo_conteudo == termo:
                vetor_tf[i] += 1
                break
    #print(vetor_tf)
    return vetor_tf

def vetorização_tf_idf(conteudo, vocabulario, vetor_ni, N):
    vetor_tf_idf = [0] * len(vocabulario)
    for termo_conteudo in conteudo:
        for i, termo in enumerate(vocabulario):
            if termo_conteudo == termo:
                vetor_tf_idf[i] += 1
                break
    for i in enumerate(vocabulario):
        if vetor_tf_idf[i] != 0:
            vetor_tf_idf[i] = (1+math.log(vetor_tf_idf[i]))*math.log(N/vetor_ni[i])
    #print(vetor_tf)
    return vetor_tf_idf