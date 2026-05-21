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


### ---------------------- ATIVIDADE 3 ------------------------ ###
import pandas as pd

###------------CRIAÇÃO DAS TABELAS COM A COMPARAÇÃO DOS MODELOS------------###
def write_results_tables(results_list, output_name, OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, output_name)

    df = pd.DataFrame(results_list)
    
    # Reorganiza para garantir que o Modelo seja a primeira coluna
    if 'Modelo' in df.columns:
        cols = ['Modelo'] + [c for c in df.columns if c != 'Modelo']
        df = df[cols]

    df.to_html(f"{path}.html", index=False, border=1)
    df.to_latex(f"{path}.tex", index=False, float_format="%.4f")
    
    print(f"Tabelas geradas em: {OUTPUT_DIR}/ ({output_name}.html, .tex)")

### ---------------------- ATIVIDADE 5 ------------------------ ###

    ###------------CRIAÇÃO DO TXT COM AS QUERIES USADAS------------###
def write_queries_file(term_queries_dict):
    OUTPUT_DIR = "data"
    nome_arquivo = "queries_2.txt"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, nome_arquivo)

    with open(path, 'w', encoding='utf-8') as f:
        # Processando term_queries_dict
        for term_query in term_queries_dict:
            category = term_query["category"]
            content = ' '.join(term_query["content"])
            
            f.write(f"{category} {content}\n")

def write_iteration_header(iteracao, nome_arquivo, OUTPUT_DIR):
    # Escreve apenas o cabeçalho da iteração
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, nome_arquivo)
    
    with open(path, 'a', encoding='utf-8') as f:
        f.write(f"{iteracao}:\n")

def write_numeric_file_2(id_consulta, ranking, nome_arquivo, OUTPUT_DIR):
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

import matplotlib.pyplot as plt

def write_metrics_plot(resultados, resultados_residuais, output_name, OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metricas = ["Precision@10", "Precision@20", "Precision@30", "MAP"]
    cores = ["blue", "green", "orange", "red"]

    iters = []
    for r in resultados:
        if isinstance(r["Iteracao"], int):
            iters.append(r["Iteracao"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Gráfico 1: Avaliação Normal ---
    ax1 = axes[0]
    for metrica, cor in zip(metricas, cores):
        valores = []
        for r in resultados:
            if isinstance(r["Iteracao"], int):
                valores.append(r[metrica])
        ax1.plot(iters, valores, marker='o', label=metrica, color=cor)

    ax1.set_title("Avaliação Normal (Rocchio)")
    ax1.set_xlabel("Iteração")
    ax1.set_ylabel("Valor")
    ax1.legend()
    ax1.grid(True)

    # --- Gráfico 2: Avaliação Residual ---
    ax2 = axes[1]
    for metrica, cor in zip(metricas, cores):
        valores_residuais = []
        for r in resultados_residuais:
            valores_residuais.append(r[metrica])
        ax2.plot(iters, valores_residuais, marker='s', linestyle='--', label=metrica, color=cor)

    ax2.set_title("Avaliação Residual")
    ax2.set_xlabel("Iteração")
    ax2.set_ylabel("Valor")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{output_name}.png")
    plt.savefig(path)
    plt.close()
    print(f"Gráfico salvo em: {path}")