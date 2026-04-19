from shared import reader, tokenizer, utils, weighting, writer, ranking

def main():
    # Ler o arquivo e gerar lista de dados brutos
    lista_documentos = reader.read_dataset_file()

    # Gerar representação vetorial
    conteudo_tokens = tokenizer.tokenize(lista_documentos) # tokenizar os arquivos 
    vocabulario = utils.criar_vocabulario(conteudo_tokens) # gerar o vocabulário
    
    N = len(conteudo_tokens) # Quantidade total de documentos do dataset
    
    # Calcula a quantidade de documentos em que o termo i aparece
    print("cálculo do ni\n")
    vetor_ni = [0] * len(vocabulario)
    for i, termo in enumerate(vocabulario):
        vetor_ni[i] = utils.ni_calculation(termo, conteudo_tokens)

    # Criar os vetores usando o vocabulário
    print("criando vetores de pesos...\n")
    vetores_tf = [] # lista de vetores TF (v1)
    vetores_idf = [] # lista de vetores IDF (v2)
    for conteudo in conteudo_tokens:
        vetores_tf.append(weighting.vetorizacao_tf_log(conteudo, vocabulario)) # v1
        vetores_idf.append(weighting.vetorizacao_idf(conteudo, vocabulario, vetor_ni, N)) # v2
    
    # ------- Gerar consultas (ou deixar as consultas salvas em um arquivo?) ------- #
    print("Gerando consultas... \n")
    term_queries = utils.get_term_queries() # 10 consultas por termos

    # Versão 1 (TF)
    term_queries_tf = [weighting.vetorizacao_tf_log(t, vocabulario) for t in term_queries] # vetoriza cada query separadamente
    doc_queries_tf, query_txt_tf = utils.get_sample_documents(vetores_tf, lista_documentos) # 50 documentos aleatórios
    queries_weights_tf = utils.unify_queries_weights(doc_queries_tf, term_queries_tf) # vetor com todas as 60 consultas
    queries_txt_tf = utils.unify_queries_txt(query_txt_tf, term_queries)

    # Versão 2 (IDF)
    term_queries_idf = [weighting.vetorizacao_idf(t, vocabulario, vetor_ni, N) for t in term_queries]  # vetoriza cada query separadamente
    doc_queries_idf, query_txt_idf = utils.get_sample_documents(vetores_idf, lista_documentos) # 50 documentos aleatórios
    queries_weights_idf = utils.unify_queries_weights(doc_queries_idf, term_queries_idf) # vetor com todas as 60 consultas
    queries_txt_idf = utils.unify_queries_txt(query_txt_idf, term_queries)

    # ------- Fazer o ranqueamento de cada consulta (retorna 30 mais similares) ------- #
    # Versão 1 (TF)
    results_tf = []
    for query in queries_weights_tf:
        results_tf.append(ranking.ranqueamento_cos(query, vetores_tf))

    # Versão 2 (IDF)
    results_idf = []
    for query in queries_weights_idf:
        results_idf.append(ranking.ranqueamento_cos(query, vetores_idf))

    # ------- Gerar os arquivos de saída ------- #
    print("Escrevendo arquivos...\n")

    # Versão 1 (TF)
    for i, query in enumerate(queries_weights_tf):
        writer.write_numeric_file(i + 1, results_tf[i], "resultados_numericos_tf_log.txt", "results/atv_3")         
        writer.write_textual_file(i, queries_txt_tf, results_tf[i], lista_documentos, "resultados_textuais_tf_log.txt", "results/atv_3") 

    # Versão 2 (IDF)
    for i, query in enumerate(queries_weights_idf):
        writer.write_numeric_file(i + 1, results_idf[i], "resultados_numericos_idf.txt", "results/atv_3")       
        writer.write_textual_file(i, queries_txt_idf, results_idf[i], lista_documentos, "resultados_textuais_idf.txt", "results/atv_3") 

if __name__ == "__main__":
    main()