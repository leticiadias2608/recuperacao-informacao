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

def write_textual_file(id_consulta, queries_txt, ranking, conteudos, nome_arquivo):
    path = os.path.join(OUTPUT_DIR, nome_arquivo)
    # pega os itens do ranking 

    docs_formatados = []
    conteudo_query = queries_txt[id_consulta]

    for doc_id in ranking:
        conteudo_ranking = conteudos[doc_id]
        trecho_ranking = conteudo_ranking[:100]
        
        docs_formatados.append(f"{trecho_ranking}")
    
    linha_arquivo = f"{conteudo_query} {docs_formatados}\n"

        # escreve linha por linha a cada iteração do loop
    with open(path, 'a', encoding='utf-8') as f:
        f.write(linha_arquivo)