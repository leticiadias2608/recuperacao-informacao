import random

###------------CRIAÇÃO DO VOCABULÁRIO------------###
def build_vocabulary(conteudo_tokens):
    vocabulario = set()
    for conteudo in conteudo_tokens:
        for token in conteudo:
            vocabulario.add(token)
    return list(vocabulario)  # retorna lista para manter a ordenação

###------------CÁLCULO DO ni------------###
def ni_calculation(termo, conteudo_tokens):
    ni = 0
    for doc in conteudo_tokens:
        for token in doc:
            if token == termo:
                ni += 1
                break
    return ni



### BUSCA POR conteúdo (2 BUSCAS POR CATEGORIA) ###

# business - profit, dollar, economy | market, bank, euro
# entertainment - oscar, award, film | comedy, actor, album
# politics - election, government, tory | debate, minister, political
# sport - victory, champion, cup | match, player, coach
# tech - microsoft, software, digital | system, technology, computer
def get_term_queries():
    dict_queries = [
    {"category": "business", "content": ["profit", "dollar", "economy"]},
    {"category": "business", "content": ["market", "bank", "euro"]},
    {"category": "entertainment", "content": ["oscar", "award", "film"]},
    {"category": "entertainment", "content": ["comedy", "actor", "album"]},
    {"category": "politics", "content": ["election", "government", "tory"]},
    {"category": "politics", "content": ["debate", "minister", "political"]},
    {"category": "sport", "content": ["victory", "champion", "cup"]},
    {"category": "sport", "content": ["match", "player", "coach"]},
    {"category": "tech", "content": ["microsoft", "software", "digital"]},
    {"category": "tech", "content": ["system", "technology", "computer"]}
]
    return dict_queries


### BUSCA POR DOCUMENTOS (50 DOCUMENTOS) ###
def get_doc_queries(lista_documentos):
    random.seed(42)

    # Selecionar 50 documentos aleatórios
    indices_sorteados = random.sample(range(len(lista_documentos)), 50)
    lista_doc_queries = []
    
    for i in indices_sorteados:
        doc = lista_documentos[i]
        
        # Dicionário da query com apenas categoria e id do documento
        doc_simplificado = {
            "category": doc["category"],
            "id": i
        }
        
        lista_doc_queries.append(doc_simplificado) # Acrescenta o dicionário na lista
     
    return lista_doc_queries


### ---------------------- ATIVIDADE 2 ------------------------ ###

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