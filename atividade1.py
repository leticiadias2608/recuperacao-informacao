from shared import reader, tokenizer, utils, weighting, writer, ranking

def main():
    # Ler o arquivo e gerar lista de dados brutos
    lista_documentos = reader.read_dataset_file()

    ### Gerar representação vetorial ###
    # Tokeinizar os arquivos e gerar o vocabulário
    conteudo_tokens = tokenizer.tokenize(lista_documentos)
    vocabulario = utils.criar_vocabulario(conteudo_tokens)
    
    N = len(conteudo_tokens) # Quantidade total de documentos do dataset
    
    # Calcula a quantidade de documentos em que o termi i aparece
    vetor_ni = [0] * len(vocabulario)
    for i, termo in enumerate(vocabulario):
        vetor_ni[i] = utils.ni_calculation(termo, conteudo_tokens)

    # Criar os vetores usando o vocabulário
    vetores_tf = [] # lista de vetores tf
    vetores_tf_idf = [] # lista de vetores (v2 com TF-IDF)
    for conteudo in conteudo_tokens:
        vetores_tf.append(weighting.vetorizacao_tf(conteudo, vocabulario)) # v1
        vetores_tf_idf.append(weighting.vetorizacao_tf_idf(conteudo, vocabulario, vetor_ni, N)) # v2
    
    # Gerar consultas (ou deixar as consultas salvas em um arquivo?)
        # Escolher 50 documentos aleatórios e 2 consultas de termo por categoria (3 termos cada)
    doc_queries_tf = utils.get_sample_documents(vetores_tf)
    term_queries = utils.get_term_queries(vocabulario)   
    # Fazer o ranqueamento de cada consulta (retorna 30 mais similares)
        # Calcular a similaridade e fazer a lista ranqueada (para v1 e para v2)
    result = ranking.ranqueamento(vetores_tf[0], vetores_tf, conteudo_tokens)

    # Gerar os arquivos de saída (para v1 e v2)


if __name__ == "__main__":
    main()