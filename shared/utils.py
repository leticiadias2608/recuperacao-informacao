
###------------FUNÇÕES GERAIS------------###

import random
###------------CÁLCULO DO ni------------###
def ni_calculation(termo, conteudo_tokens):
    ni = 0
    for doc in conteudo_tokens:
        for token in doc:
            if token == termo:
                ni += 1
                break
    return ni

###------------CRIAÇÃO DO VOCABULÁRIO------------###
def criar_vocabulario(conteudo_tokens):
    vocabulario = set()
    for conteudo in conteudo_tokens:
        for token in conteudo:
            vocabulario.add(token)
    return list(vocabulario)  # retorna lista para manter a ordenação

###------------CRIAÇÃO DAS BUSCAS------------###

### 50 DOCUMENTOS ALEATÓRIOS PARA BUSCA ###
def get_sample_documents(vetores_tf, lista_documentos):
    conteudos = []
    for conteudo in lista_documentos:
        conteudos.append(conteudo['content'])
        
    random.seed(42)
    doc_queries = []
    query_txt = []
    doc_ids = random.sample(range(len(vetores_tf)), k=50)
    for id in doc_ids:
        doc_queries.append(vetores_tf[id])
        query_txt.append(conteudos[id])
   
    #print(doc_queries)
    return doc_queries, query_txt

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

### UNIR AS 60 BUSCAS (DOCUMENTOS E TERMOS) EM UMA ÚNICA LISTA ###
def unify_queries_weights(doc_queries, term_queries):
    for query in term_queries:
        doc_queries.append(query)

    return doc_queries

def unify_queries_txt(doc_queries_txt, term_queries_txt):
    for query in term_queries_txt:
        doc_queries_txt.append(query)

    return doc_queries_txt