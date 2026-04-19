
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
    doc_queries = {} # {category: sport, query: []}
    query_txt = []
    doc_ids = random.sample(range(len(vetores_tf)), k=50)
    
    for id, conteudo in doc_ids, lista_documentos:
        categoria = conteudo["categoria"]
        doc_queries[id] = {
            "category": categoria,
            "query": vetores_tf[id]
        }
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
    # term_queries no formato {category: query}
    term_queries = {"business": ["profit", "dollar", "economy"], 
                    "business": ["market", "bank", "euro"], 
                    "entertainment": ["oscar", "award", "film"],
                    "entertainment": ["comedy", "actor", "album"],
                    "politics": ["election", "government", "tory"],
                    "politics": ["debate", "minister", "political"],
                    "sport": ["victory", "champion", "cup"],
                    "sport": ["match", "player", "coach"],
                    "tech": ["microsoft", "software", "digital"],
                    "tech": ["system", "technology", "computer"]}

    return term_queries

### UNIR AS 60 BUSCAS (DOCUMENTOS E TERMOS) EM UMA ÚNICA LISTA ###
def unify_queries_weights(doc_queries, term_queries):
    for query in term_queries:
        doc_queries.update({"category": query["category"], "query": query["query"]})

    return doc_queries

def unify_queries_txt(doc_queries_txt, term_queries_txt):
    for query in term_queries_txt:
        doc_queries_txt.append(query)

    return doc_queries_txt

### ATIVIDADE 2 ###

def average_doclen(documents): # recebe conteudo_tokens
    sum = 0 # é necessario?
    for document in documents:
        sum += len(document)
    avg = sum/len(documents)
    return avg

def B_frequency(K, b, f, avg_doclen, d):
    aux = ((1-b)+b*(len(d)/avg_doclen))
    B_f = (K+1)*f/(K*aux+f)
    return B_f

def get_term_frequency(documento_tokens, vocabulario):
    f = []
    for k in vocabulario:
        f.append(documento_tokens.count(k))
    return f