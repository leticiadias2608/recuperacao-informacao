###------------RESPONSÁVEL POR ESCREVER ARQUIVOS DE SAÍDA E SALVAR DADOS------------###
import os 
from shared import tokenizer

###------------CRIAÇÃO DO TXT COM AS QUERIES USADAS------------###
def write_queries_file(term_queries_dict, doc_queries_dict, conteudo_tokens):
    OUTPUT_DIR = "data"
    nome_arquivo = "queries.txt"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, nome_arquivo)

    with open(path, 'w', encoding='utf-8') as f:
        for doc in doc_queries_dict:
            doc_index = doc["id"]
            category = doc["category"]
            tokens = conteudo_tokens[doc_index]
            content = ' '.join(tokens)
            
            f.write(f"{category} {content}\n")

        # 2. Processando term_queries_dict (no final do arquivo)
        for term_query in term_queries_dict:
            category = term_query["category"]
            content = ' '.join(term_query["content"])
            
            f.write(f"{category} {content}\n")

###------------CRIAÇÃO DO TXT COM RESULTADO NUMÉRICOS------------###
def write_numeric_file(id_consulta, ranking, nome_arquivo, OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)  
    docs_formatados = []
    path = os.path.join(OUTPUT_DIR, nome_arquivo)
    
    # pega os itens do ranking 
    for doc_id in ranking:
        docs_formatados.append(f"{doc_id}")
        
    # junta todos os ids da lista com espaço 
    string_documentos = " ".join(docs_formatados)
    linha_arquivo = f"{id_consulta} {string_documentos}\n" # formato: query_id doc_i doc_j doc_z ...
    
    # escreve linha por linha a cada iteração do loop
    with open(path, 'a', encoding='utf-8') as f:
        f.write(linha_arquivo)


###------------CRIAÇÃO DO TXT COM RESULTADOS TEXTUAIS------------###
def write_textual_file(id_consulta, queries_txt, ranking, lista_documentos, nome_arquivo, OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, nome_arquivo)

    TRECHO = 100
    conteudo_query = queries_txt[:20]
    # exibe só trecho da consulta se for texto longo (doc query), ou os termos diretamente
    if isinstance(conteudo_query, list):
        trecho_query = ' '.join(conteudo_query)
    else:
        trecho_query = conteudo_query

    linhas = [f'"{trecho_query}"']
    for rank, doc_id in enumerate(ranking, start=1):
        conteudo_doc = lista_documentos[doc_id - 1]['content'] 
        linhas.append(f'{rank} "{conteudo_doc[:TRECHO]}"')

    with open(path, 'a', encoding='utf-8') as f:
        f.write('\n'.join(linhas) + '\n\n')