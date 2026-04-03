###------------FUNÇÕES DE GERAÇÃO VETORES DE PESOS------------###

### VETORIZAÇÃO ###
def vetorizacao_tf(conteudo, vocabulario):
    vetor_tf = [0] * len(vocabulario)
    for termo_conteudo in conteudo:
        for i, termo in enumerate(vocabulario):
            if termo_conteudo == termo:
                vetor_tf[i] += 1
    #print(vetor_tf)
    return vetor_tf