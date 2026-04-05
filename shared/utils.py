
###------------FUNÇÕES GERAIS------------###

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

### 50 DOCUMENTOS ALEATÓRIOS PARA BUSCA ###
def get_sample_documents(vetores_tf):
    doc_queries = random.choices(vetores_tf, k=50)

    print(doc_queries)
    return doc_queries

### BUSCA POR TERMOS (2 BUSCAS POR CATEGORIA) ###

# business - profit, dollar, economy | market, bank, euro
# entertainment - oscar, award, film | comedy, actor, album
# politics - election, government, tory | debate, minister, political
# sport - victory, champion, cup | match, player, coach
# tech - microsoft, software, digital | system, technology, computer
def get_term_queries():
    term_queries = [["profit", "dollar", "economy"], ["market", "bank", "euro"], 
                    ["oscar", "award", "film"],["comedy", "actor", "album"],
                    ["election", "government", "tory"], ["debate", "minister", "political"],
                    ["victory", "champion", "cup"], ["match", "player", "coach"],
                    ["microsoft", "software", "digital"], ["system", "technology", "computer"]]

    return term_queries

def unify_queries(doc_queries_tf, term_queries_tf):
    for query in term_queries_tf:
        doc_queries_tf.append(query)

    return doc_queries_tf