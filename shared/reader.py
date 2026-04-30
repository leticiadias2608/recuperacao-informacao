###------------RESPONSÁVEL POR LER ARQUIVOS E DADOS SALVOS------------###

### LEITURA DO ARQUIVO (DATASET) ###
def read_dataset_file():
    dataset_path = "data/bbc-news-data.csv" # caminho para o dataset

    with open(dataset_path, 'r', encoding='utf-8') as arquivo:
        linhas = arquivo.readlines()
        
        cabecalho = linhas[0].lower().strip().split('\t')

        
        lista_documentos = []
        for linha in linhas[1:]:
            valores = linha.lower().strip().split('\t')
            documento = dict(zip(cabecalho, valores))

            lista_documentos.append(documento) # lista de dicionario

        return lista_documentos

### LEITURA DO ARQUIVO DE CONSULTAS ###
def read_queries_file():
    dataset_path = "data/queries.txt" # caminho para o dataset

    with open(dataset_path, 'r', encoding='utf-8') as arquivo:
        linhas = arquivo.readlines()
        
        queries_dict = []
        for linha in linhas:
            palavras = linha.strip().split(' ')

            if not palavras or palavras == ['']:
                continue
                
            categoria_lida = palavras[0] # A primeira palavra (índice 0) é a categoria
            termos_lidos = palavras[1:] # Da segunda palavra em diante (índice 1 até o final), são os termos
            
            # Remonta o dicionário e adiciona na lista
            queries_dict.append({
                "category": categoria_lida,
                "content": termos_lidos
            })

        return queries_dict
    

### LEITURA DAS RANKED LISTS ###
def read_rankeds_file(dataset_path):

    with open(dataset_path, 'r', encoding='utf-8') as arquivo:
        linhas = arquivo.readlines()
        
        ranked_lists = []
        for linha in linhas:
            palavras = linha.strip().split(' ')

            if not palavras or palavras == ['']:
                continue
                
            query_lida = palavras[0] # A primeira palavra (índice 0) é a query
            ranking_lido = palavras[1:] # Da segunda palavra em diante (índice 1 até o final), são os documentos rankeados
            
            # Remonta o dicionário e adiciona na lista
            ranked_lists.append({
                "query": query_lida,
                "ranking": ranking_lido
            })
        return ranked_lists
    