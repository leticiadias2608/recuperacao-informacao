from shared import reader, tokenizer, utils, weighting, writer, ranking

def main():
    # Ler o arquivo e gerar lista de dados brutos
    lista_documentos = reader.read_dataset_file()

    ### Gerar representação vetorial ###
    # Tokeinizar os arquivos e gerar o vocabulário
    conteudo_tokens = tokenizer.tokenize(lista_documentos)
    vocabulario = utils.criar_vocabulario(conteudo_tokens)

    # Criar os vetores usando o vocabulário (v1 com TF)
    vetores_tf = [] # lista de vetores tf
    for conteudo in conteudo_tokens:
        vetores_tf.append(weighting.vetorizacao_tf(conteudo, vocabulario)) 
        # Criar os vetores usando o vocabulário (v2 com TF-IDF)
    
    # Gerar consultas (ou deixar as consultas salvas em um arquivo?)
        # Escolher 50 documentos aleatórios e 2 consultas de termo por categoria (3 termos cada)
    utils.sample_documents(vetores_tf)

    # Fazer o ranqueamento de cada consulta (retorna 30 mais similares)
        # Calcular a similaridade e fazer a lista ranqueada (para v1 e para v2)
    result = ranking.ranqueamento(vetores_tf[0], vetores_tf, conteudo_tokens)

    # Gerar os arquivos de saída (para v1 e v2)


if __name__ == "__main__":
    main()