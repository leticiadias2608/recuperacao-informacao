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

def build_index(conteudo_tokens):
    vocabulario = build_vocabulary(conteudo_tokens)
    vetor_ni = [ni_calculation(t, conteudo_tokens) for t in vocabulario]
    ni_map = {t: vetor_ni[i] for i, t in enumerate(vocabulario)}
    avg_dl = average_doclen(conteudo_tokens)
    return vocabulario, vetor_ni, ni_map, avg_dl

### ---------------------- ATIVIDADE 5 ------------------------ ###

### BUSCA POR conteúdo (2 BUSCAS POR CATEGORIA) ###

# business - economy | market, bank
# entertainment - award | actor, album
# politics - election | political, tory
# sport - player  | match, coach
# tech - computer | microsoft, software 
""" def get_term_queries_2():
    dict_queries = [
    {"category": "business", "content": ["economy"]},
    {"category": "business", "content": ["market", "bank"]},
    {"category": "entertainment", "content": ["award"]},
    {"category": "entertainment", "content": ["actor", "album"]},
    {"category": "politics", "content": ["election"]},
    {"category": "politics", "content": ["political", "tory"]},
    {"category": "sport", "content": ["player"]},
    {"category": "sport", "content": ["match", "coach"]},
    {"category": "tech", "content": ["computer"]},
    {"category": "tech", "content": ["microsoft", "software"]}
    ]
    return dict_queries """

""" def get_term_queries_2():
    dict_queries = [
    {"category": "business", "content": ["market"]},
    {"category": "business", "content": ["firm", "chief"]},
    {"category": "entertainment", "content": ["music"]},
    {"category": "entertainment", "content": ["star", "show"]},
    {"category": "politics", "content": ["government"]},
    {"category": "politics", "content": ["plans", "general"]},
    {"category": "sport", "content": ["game"]},
    {"category": "sport", "content": ["play", "final"]},
    {"category": "tech", "content": ["service"]},
    {"category": "tech", "content": ["internet", "using"]},
    ]
    return dict_queries """

def get_term_queries_2():
    dict_queries = [
    # business: 'world' e 'down' — aparecem em todas as categorias igualmente
    {"category": "business", "content": ["world"]},
    {"category": "business", "content": ["down", "much"]},

    # entertainment: 'number' e 'four' — genéricos, sem vínculo claro
    {"category": "entertainment", "content": ["number"]},
    {"category": "entertainment", "content": ["four", "five"]},

    # politics: 'added' e 'next' — verbos/adjetivos completamente genéricos
    {"category": "politics", "content": ["added"]},
    {"category": "politics", "content": ["next", "week"]},

    # sport: 'since' e 'still' — praticamente aleatórios entre categorias
    {"category": "sport", "content": ["since"]},
    {"category": "sport", "content": ["still", "should"]},

    # tech: 'made' e 'take' — os mais distribuídos do dataset
    {"category": "tech", "content": ["made"]},
    {"category": "tech", "content": ["take", "next"]},
    ]
    return dict_queries

# Remove dos rankings os documentos que já foram vistos em iterações anteriores.
def filter_seen_docs(ranked_lists_iter, docs_vistos):
    filtered = []
    for rank in ranked_lists_iter:
        ranking_filtrado = [doc for doc in rank["ranking"] if doc not in docs_vistos]
        filtered.append({
            "query": rank["query"],
            "ranking": ranking_filtrado
        })
    return filtered

