import time
from shared import reader, tokenizer, utils, weighting, writer, ranking, evaluation

def main():
    # Ler o arquivo e gerar lista de dados brutos
    lista_documentos = reader.read_dataset_file()
    queries_dict     = reader.read_queries_file("data/queries.txt") 
    
    # ---- sem pré-processamento ---- #
    conteudo_tokens = tokenizer.tokenize(lista_documentos) # tokenizar os arquivos 
    N = len(conteudo_tokens) # Quantidade total de documentos do dataset
    query_tokens_base = []
    for q in queries_dict:
        query_tokens_base.append(q["content"])
        
    # ---- pré-processamento ---- #
    # Apenas stopwords
    conteudo_sw    = tokenizer.remove_stop_words(conteudo_tokens)
    query_sw       = tokenizer.remove_stop_words(query_tokens_base)
    
    # Apenas stemming
    conteudo_stem  = tokenizer.stemming(conteudo_tokens)
    query_stem     = tokenizer.stemming(query_tokens_base)

    # Stopwords e Stemming
    conteudo_sw_stem = tokenizer.stemming(conteudo_sw)
    query_sw_stem    = tokenizer.stemming(query_sw)
    
    variantes = [
        ("sem_preprocessamento", conteudo_tokens,    query_tokens_base),
        ("stopwords",            conteudo_sw,         query_sw),
        ("stemming",             conteudo_stem,        query_stem),
        ("stopwords_stemming",   conteudo_sw_stem,     query_sw_stem),
    ]
    
    resultados_totais = []
    
    for nome_variante, conteudo_tokenizado, queries in variantes:
        
        path_saida = "results/atv_4"
        
        # TF IDF
        nome_num_tf_idf   = f"resultados_numericos_TF-IDF_{nome_variante}.txt"
        nome_txt_tf_idf    = f"resultados_textuais_TF-IDF_{nome_variante}.txt"
        vocabulario, vetor_ni, ni_map, avg_dl = utils.build_index(conteudo_tokenizado)
        
        vetores_tf_idf = []
        queries_tf_idf = []
        
        tempo_inicial_tf_idf = time.time()
        for conteudo in conteudo_tokenizado:
            vetores_tf_idf.append(weighting.vetorizacao_tf_idf(conteudo, vocabulario, vetor_ni, N)) 
        for query in queries:
            queries_tf_idf.append(weighting.vetorizacao_tf_idf(query, vocabulario, vetor_ni, N)) 
        
        for i, query_vec in enumerate(queries_tf_idf):
            result_tf_idf = ranking.ranqueamento_cos(query_vec, vetores_tf_idf)
            writer.write_numeric_file(i + 1, result_tf_idf, nome_num_tf_idf, path_saida)
            writer.write_textual_file(i, queries[i], result_tf_idf, lista_documentos, nome_txt_tf_idf, path_saida)
        tempo_final_tf_idf = time.time() - tempo_inicial_tf_idf

        # BM25
        nome_num_bm25    = f"resultados_numericos_BM25_{nome_variante}.txt"
        nome_txt_bm25    = f"resultados_textuais_BM25_{nome_variante}.txt"
        
        tempo_inicial_bm25 = time.time()
        for i, query in enumerate(queries):
            result_bm25 = ranking.ranqueamento_prob(query, conteudo_tokenizado, ni_map, N, avg_dl, func_sim="BM25", K=1.5, b=0.75)
            writer.write_numeric_file(i + 1, result_bm25, nome_num_bm25, path_saida)
            writer.write_textual_file(i, queries[i], result_bm25, lista_documentos, nome_txt_bm25, path_saida)
        tempo_final_bm25 = time.time() - tempo_inicial_bm25    
            
        # Avaliação TF-IDF
        ranked_lists_tfidf = reader.read_rankeds_file(f"./{path_saida}/{nome_num_tf_idf}")
        metricas_tfidf = evaluation.evaluate(queries_dict, ranked_lists_tfidf, lista_documentos)
        metricas_tfidf["Modelo"] = f"TF-IDF_{nome_variante}"
        metricas_tfidf["Tamanho"] = len(vocabulario)
        metricas_tfidf["Tempo de Execução"] = tempo_final_tf_idf
        resultados_totais.append(metricas_tfidf)

        # Avaliação BM25
        ranked_lists_bm25 = reader.read_rankeds_file(f"./{path_saida}/{nome_num_bm25}")
        metricas_bm25 = evaluation.evaluate(queries_dict, ranked_lists_bm25, lista_documentos)
        metricas_bm25["Modelo"] = f"BM25_{nome_variante}"
        metricas_bm25["Tamanho"] = len(vocabulario)
        metricas_bm25["Tempo de Execução"] = tempo_final_bm25
        resultados_totais.append(metricas_bm25)

    print(resultados_totais)
    writer.write_results_tables(resultados_totais, "comparacao_modelos", "results/atv_4")

if __name__ == "__main__":
    main()