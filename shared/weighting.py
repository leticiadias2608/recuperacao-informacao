###------------FUNÇÕES DE GERAÇÃO VETORES DE PESOS------------###
import math

### VETORIZAÇÃO ###
def vetorizacao_tf(conteudo, vocabulario):
    vocab_idx = {termo: i for i, termo in enumerate(vocabulario)}  # dict uma vez só
    vetor_tf = [0] * len(vocabulario)
    for termo_conteudo in conteudo:
        if termo_conteudo in vocab_idx:
            vetor_tf[vocab_idx[termo_conteudo]] += 1
    return vetor_tf

def vetorizacao_tf_idf(conteudo, vocabulario, vetor_ni, N):
    vocab_idx = {termo: i for i, termo in enumerate(vocabulario)}
    vetor_tf_idf = [0] * len(vocabulario)
    for termo_conteudo in conteudo:
        if termo_conteudo in vocab_idx:
            vetor_tf_idf[vocab_idx[termo_conteudo]] += 1
    for i in range(len(vocabulario)):
        if vetor_tf_idf[i] != 0:
            vetor_tf_idf[i] = (1 + math.log(vetor_tf_idf[i])) * math.log(N / vetor_ni[i])
    return vetor_tf_idf