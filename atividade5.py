from shared import reader, tokenizer, utils, weighting, writer, ranking, evaluation, feedback

def main():
    # Ler o arquivo e gerar lista de dados brutos
    lista_documentos = reader.read_dataset_file()
    
    # Gerar representação vetorial
    conteudo_tokens = tokenizer.tokenize(lista_documentos) # tokenizar os arquivos 
    vocabulario = utils.build_vocabulary(conteudo_tokens) # construir o vocabulário
    
    N = len(conteudo_tokens) # Quantidade total de documentos do dataset
    
    # Calcula a quantidade de documentos em que o termo i aparece
    print("Cálculo do ni\n")
    vetor_ni = [0] * len(vocabulario)
    for i, termo in enumerate(vocabulario):
        vetor_ni[i] = utils.ni_calculation(termo, conteudo_tokens)

    # Gerar consultas e salvá-las no arquivo
    print("Gerando consultas... \n")
    term_queries_dict = utils.get_term_queries_2() # 10 consultas por termos
    writer.write_queries_file(term_queries_dict) # Escreve txt com as queries

    ##-------------------------------------------------------------------------##
    ## ---------------- Criar os vetores usando o vocabulário ---------------- ##
    ##-------------------------------------------------------------------------##
    # Vetorização dos documentos
    print("Criando vetores de pesos...\n")
    vetores_tf_idf = [] # lista de vetores
    for conteudo in conteudo_tokens:
        vetores_tf_idf.append(weighting.vetorizacao_tf_idf(conteudo, vocabulario, vetor_ni, N))
    
    # Vetorização das consultas
    queries_dict = reader.read_queries_file("data/queries_2.txt") # Dicionário com todas as consultas no formato {categoria, conteudo_token}

    queries_tf_idf = []
    for query in queries_dict:
        queries_tf_idf.append(weighting.vetorizacao_tf_idf(query["content"], vocabulario, vetor_ni, N)) 

    ##-----------------------------------------------------------------------------##
    ## ---- Fazer o ranqueamento de cada consulta (retorna 50 mais similares) ---- ##
    ##-----------------------------------------------------------------------------##
    print("Escrevendo arquivos...\n")

    for i, query in enumerate(queries_tf_idf):
        results_tf_idf = ranking.ranqueamento_cos(query, vetores_tf_idf, 50)
        # Escrevendo arquivos de resultado
        writer.write_numeric_file(i + 1, results_tf_idf, "resultados_originais_TF-IDF.txt", "results/atv_5") 
    
    resultados = []
    resultados_residuais = []

    nome_arquivo = f"./results/atv_5/resultados_originais_TF-IDF.txt"  
    ranked_lists_originais =  reader.read_ranked_file(nome_arquivo)
    metricas = evaluation.evaluate_2(queries_dict, ranked_lists_originais, lista_documentos)
    metricas["Iteracao"] = "original"
    resultados.append(metricas)  

    alfa = 0.8
    beta = 0.75
    gama = 0.15

    ranked_lists = ranked_lists_originais
    queries_atuais = queries_tf_idf # começa com as queries originais

    for i in range(10):
        queries_modificadas = []  # reinicia a cada iteração do Rocchio
        for j, rank in enumerate(ranked_lists):
            query = queries_atuais[j] # usa o vetor da query atual
            
            doc_rel, doc_n = feedback.feedback(rank, lista_documentos, vetores_tf_idf, queries_dict)

            # cálculo da query modificada pelo método rocchio 
            qm = feedback.rocchio_method(alfa, beta, gama, query, doc_rel, len(doc_rel), doc_n, len(doc_n), vocabulario)

            # Monta o dicionário da query modificada
            queries_modificadas.append({
                "category": queries_dict[j]["category"],  # categoria da query original
                "content": qm                              # vetor ponderado pelo Rocchio
            })
        
        print("Escrevendo arquivos...\n")
        # Escrita do cabeçalho da iteração no arquivo
        writer.write_iteration_header(i, "resultados_modificados_TF-IDF.txt", "results/atv_5")

        for j, query in enumerate(queries_modificadas):
            results_tf_idf = ranking.ranqueamento_cos(query['content'], vetores_tf_idf, 50)
            # Escrita dos resultados no arquivo
            writer.write_numeric_file_2(j + 1, results_tf_idf, "resultados_modificados_TF-IDF.txt", "results/atv_5")

        # atualiza ranked_lists 
        ranked_lists = reader.read_rankeds_file_2(f"./results/atv_5/resultados_modificados_TF-IDF.txt")[i]
        
        # salva as qm desta iteração
        """ for q in queries_modificadas:
            queries_atuais.append(q['content']) """
        queries_atuais = [q["content"] for q in queries_modificadas]  

    # Conjunto dos documentos já vistos
    docs_vistos = set()
    for rank in ranked_lists_originais:        
        for doc_id in rank["ranking"]:        
            docs_vistos.add(doc_id)   

    nome_arquivo = f"./results/atv_5/resultados_modificados_TF-IDF.txt"  
    ranks =  reader.read_rankeds_file_2(nome_arquivo)

    for i in range(10):
        ranked_lists = ranks[i]

        metricas = evaluation.evaluate_2(queries_dict, ranked_lists, lista_documentos)
        metricas["Iteracao"] = i
        resultados.append(metricas) 

        ranked_residual = utils.filter_seen_docs(ranked_lists, docs_vistos)
        metricas_residual = evaluation.evaluate_2(queries_dict, ranked_residual, lista_documentos)
        metricas_residual["Iteracao"] = i
        resultados_residuais.append(metricas_residual)

        # ← NOVO: acumula docs vistos desta iteração para a próxima
        for rank in ranked_lists:
            docs_vistos.update(rank["ranking"])


    print("Escrevendo tabela de resultados...")
    print(resultados)
    writer.write_results_tables(resultados, "comparacao_modelos", "results/atv_5")
    writer.write_results_tables(resultados_residuais, "comparacao_residual", "results/atv_5")
    writer.write_metrics_plot(resultados, resultados_residuais, "graficos_metricas", "results/atv_5")

if __name__ == "__main__":
    main()