from shared import reader, tokenizer, utils, weighting, writer, ranking

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
    term_queries_dict = utils.get_term_queries() # 10 consultas por termos
    doc_queries_dict = utils.get_doc_queries(lista_documentos) # 50 documentos aleatórios
    writer.write_queries_file(term_queries_dict, doc_queries_dict, conteudo_tokens) # Escreve txt com as queries

    ##-------------------------------------------------------------------------##
    ## ---------------- Criar os vetores usando o vocabulário ---------------- ##
    ##-------------------------------------------------------------------------##
    # Vetorização dos documentos
    print("Criando vetores de pesos...\n")
    vetores_tf = [] # lista de vetores TF (v1)
    vetores_idf = [] # lista de vetores IDF (v1)
    vetores_tf_idf = [] # lista de vetores (v2)
    for conteudo in conteudo_tokens:
        vetores_tf.append(weighting.vetorizacao_tf_log(conteudo, vocabulario)) # v1
        vetores_idf.append(weighting.vetorizacao_idf(conteudo, vocabulario, vetor_ni, N)) # v1
        vetores_tf_idf.append(weighting.vetorizacao_tf_idf(conteudo, vocabulario, vetor_ni, N)) # v2
    
    # Vetorização das consultas
    queries_dict = reader.read_queries_file() # Dicionário com todas as consultas no formato {categoria, conteudo_token}

    queries_tf = []
    queries_idf = []
    queries_tf_idf = []
    for query in queries_dict:
        queries_tf.append(weighting.vetorizacao_tf_log(query["content"], vocabulario)) # v1
        queries_idf.append(weighting.vetorizacao_idf(query["content"], vocabulario, vetor_ni, N)) # v1
        queries_tf_idf.append(weighting.vetorizacao_tf_idf(query["content"], vocabulario, vetor_ni, N)) # v2

    ##-----------------------------------------------------------------------------##
    ## ---- Fazer o ranqueamento de cada consulta (retorna 30 mais similares) ---- ##
    ##-----------------------------------------------------------------------------##
    print("Escrevendo arquivos...\n")


    for i, query in enumerate(queries_tf):
        results_tf = ranking.ranqueamento_cos(query, vetores_tf)
        # Escrevendo arquivos de resultado
        writer.write_numeric_file(i + 1, results_tf, "resultados_numericos_tf.txt", "results/atv_1")
        writer.write_textual_file(i, queries_dict[i]["content"], results_tf[i], lista_documentos, "resultados_textuais_tf.txt", "results/atv_1") 
    
    for i, query in enumerate(queries_idf):
        results_idf = ranking.ranqueamento_cos(query, vetores_idf)
        # Escrevendo arquivos de resultado
        writer.write_numeric_file(i + 1, results_idf, "resultados_numericos_idf.txt", "results/atv_1")
        writer.write_textual_file(i, queries_dict[i]["content"], results_idf[i], lista_documentos, "resultados_textuais_idf.txt", "results/atv_1") 
    
    for i, query in enumerate(queries_tf_idf):
        results_tf_idf = ranking.ranqueamento_cos(query, vetores_tf_idf)
        # Escrevendo arquivos de resultado
        writer.write_numeric_file(i + 1, results_tf_idf, "resultados_numericos_tf_idf.txt", "results/atv_1")
        writer.write_textual_file(i, queries_dict[i]["content"], results_tf_idf[i], lista_documentos, "resultados_textuais_tf_idf.txt", "results/atv_1") 

if __name__ == "__main__":
    main()