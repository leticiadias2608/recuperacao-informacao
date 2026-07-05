###------------RESPONSÁVEL POR LER ARQUIVOS E DADOS SALVOS------------###

import numpy as np

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
def read_queries_file(dataset_path):

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
def read_ranked_file(dataset_path):

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
    
def read_rankeds_file_2(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as arquivo:
        linhas = arquivo.readlines()

    ranked_lists = {}  # dicionário: {iteracao: [lista de ranks]}
    iteracao_atual = None

    for linha in linhas:
        stripped = linha.strip()

        if not stripped:
            continue

        if stripped.endswith(":") and stripped[:-1].isdigit():
            iteracao_atual = int(stripped[:-1])
            if iteracao_atual not in ranked_lists:
                ranked_lists[iteracao_atual] = []  # inicializa a sublista da iteração
            continue

        if iteracao_atual is not None:
            palavras = stripped.split(' ')
            query_lida = palavras[0]
            ranking_lido = palavras[1:]
            ranked_lists[iteracao_atual].append({
                "query": query_lida,
                "ranking": ranking_lido
            })

    return ranked_lists

def read_features_file():
    dataset_path = "./data/features/features.txt"
    with open(dataset_path, 'r', encoding='utf-8') as arquivo:
        linhas = arquivo.readlines()

    features_list = []
    for linha in linhas:
        # 1. Divide a string em uma lista de strings individuais
        partes = linha.split()
        
        # 2. Converte a lista de strings para um array NumPy de floats
        vetor_numerico = np.array(partes, dtype=np.float32)
        
        # 3. Adiciona o vetor convertido à nossa lista final
        features_list.append(vetor_numerico)

    return features_list


def read_metadata():
    dataset_path = "./data/features/metadata.txt"

    with open(dataset_path, 'r', encoding='utf-8') as arquivo:
        linhas = arquivo.readlines()
        
    queries_dict = []
    for linha in linhas:
        palavras = linha.strip().split(' ')

        if not palavras or palavras == ['']:
            continue
            
        query = palavras[0] # A primeira palavra (índice 0) é a query
        label = palavras[1] # A segunda palavra é o rótulo
        classe = palavras[2] # A terceira palavra é a classe
        
        # Remonta o dicionário e adiciona na lista
        queries_dict.append({
            "query_id": query,
            "label": label,
            "category": classe
        })
    return queries_dict