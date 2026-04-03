import random
###------------CRIAÇÃO DO VOCABULÁRIO------------###
def criar_vocabulario(conteudo_tokens):
    vocabulario = set()
    for conteudo in conteudo_tokens:
        for token in conteudo:
            #print(document['content'])
            vocabulario.add(token)
    #print(vocabulario)
    return vocabulario

###------------CRIAÇÃO DAS BUSCAS------------###
def sample_documents(vetores_tf):
    query_docs = random.choices(vetores_tf, k=50)

    print(query_docs)
    return query_docs