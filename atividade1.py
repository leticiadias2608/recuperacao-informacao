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
    vetores_tf_idf = [] # lista de vetores (v2)
    for conteudo in conteudo_tokens:
        vetores_tf.append(weighting.vetorizacao_tf(conteudo, vocabulario)) # v1
        vetores_tf_idf.append(weighting.vetorizacao_tf_idf(conteudo, vocabulario, vetor_ni, N)) # v2
    
    # ------- Gerar consultas (ou deixar as consultas salvas em um arquivo?) ------- #
    print("Gerando consultas... \n")
    term_queries = utils.get_term_queries() # 10 consultas por termos

    # Versão 1 (TF)
    term_queries_v1 = [weighting.vetorizacao_tf(t, vocabulario) for t in term_queries]  # CORRIGIDO: vetoriza cada query separadamente
    doc_queries_tf, query_txt_v1 = utils.get_sample_documents(vetores_tf, lista_documentos)
    queries_weights_v1 = utils.unify_queries_weights(doc_queries_tf, term_queries_v1)
    queries_txt_v1 = utils.unify_queries_txt(query_txt_v1, term_queries)

    # Versão 2 (TF-IDF)
    term_queries_v2 = [weighting.vetorizacao_tf_idf(t, vocabulario, vetor_ni, N) for t in term_queries]  # CORRIGIDO
    doc_queries_tf_idf, query_txt_v2 = utils.get_sample_documents(vetores_tf_idf, lista_documentos)
    queries_weights_v2 = utils.unify_queries_weights(doc_queries_tf_idf, term_queries_v2)
    queries_txt_v2 = utils.unify_queries_txt(query_txt_v2, term_queries)

    # ------- Fazer o ranqueamento de cada consulta (retorna 30 mais similares) ------- #
    # Versão 1 (TF)
    results_v1 = []
    for query in queries_weights_v1:
        results_v1.append(ranking.ranqueamento(query, vetores_tf, conteudo_tokens))

    # Versão 2 (TF-IDF)
    results_v2 = []
    for query in queries_weights_v2:
        results_v2.append(ranking.ranqueamento(query, vetores_tf_idf, conteudo_tokens))

    # ------- Gerar os arquivos de saída ------- #
    print("Escrevendo arquivos...\n")

    # Versão 1 (TF)
    # Versão 1 (TF)
    for i, query in enumerate(queries_weights_v1):
        writer.write_numeric_file(i + 1, results_v1[i], "resultados_numericos_v1.txt")         # CORRIGIDO: results_v1[i]
        writer.write_textual_file(i, queries_txt_v1, results_v1[i], lista_documentos, "resultados_textuais_v1.txt")  # CORRIGIDO: results_v1[i]

    # Versão 2 (TF-IDF)
    for i, query in enumerate(queries_weights_v2):
        writer.write_numeric_file(i + 1, results_v2[i], "resultados_numericos_v2.txt")         # CORRIGIDO
        writer.write_textual_file(i, queries_txt_v2, results_v2[i], lista_documentos, "resultados_textuais_v2.txt")  # CORRIGIDO

if __name__ == "__main__":
    main()