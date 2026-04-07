###------------RESPONSÁVEL POR ESCREVER ARQUIVOS DE SAÍDA E SALVAR DADOS------------###
import os 

OUTPUT_DIR = "results"

""" def write_numeric_file(id_consulta, ranking_tokens, conteudo_tokens):
    docs_formatados = []
    nome_arquivo = "resultados_numericos_v1.txt"
    path = os.path.join(OUTPUT_DIR, nome_arquivo)
    
    # pega os 30 primeiros itens do ranking 
    for doc_tokens in ranking_tokens[:30]:
        # procura a posição daquele documento em conteudo_tokens
        doc_id = conteudo_tokens.index(doc_tokens)
        docs_formatados.append(f"{doc_id}")
        
    # junta todos os ids da lista com espaço 
    string_documentos = " ".join(docs_formatados)
    linha_arquivo = f"{id_consulta} {string_documentos}\n" # formato: consulta doc1 doc2 doc3 ...
    
    # escreve a linha por linha a cada iteração do loop
    with open(path, 'a', encoding='utf-8') as f:
        f.write(linha_arquivo) """

def write_numeric_file(id_consulta, ranking, nome_arquivo):
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # ADICIONAR ESSA LINHA
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

def write_textual_file(id_consulta, queries_txt, ranking, lista_documentos, nome_arquivo):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, nome_arquivo)

    TRECHO = 100
    conteudo_query = queries_txt[id_consulta]
    # exibe só trecho da consulta se for texto longo (doc query), ou os termos diretamente
    if isinstance(conteudo_query, list):
        trecho_query = ' '.join(conteudo_query)
    else:
        trecho_query = conteudo_query[:TRECHO]

    linhas = [f'"{trecho_query}"']
    for rank, doc_id in enumerate(ranking, start=1):
        conteudo_doc = lista_documentos[doc_id - 1]['content']  # CORRIGIDO: doc_id é 1-based
        linhas.append(f'{rank} "{conteudo_doc[:TRECHO]}"')

    with open(path, 'a', encoding='utf-8') as f:
        f.write('\n'.join(linhas) + '\n\n')