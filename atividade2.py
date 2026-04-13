from shared import reader, tokenizer, utils, writer, ranking

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

    # Converte vetor_ni (lista) para dicionário {termo: ni}
    ni_map = {termo: vetor_ni[i] for i, termo in enumerate(vocabulario)}
    
    # ------- Gerar consultas (ou deixar as consultas salvas em um arquivo?) ------- #
    print("Gerando consultas... \n")
    term_queries = utils.get_term_queries() # 10 consultas por termos

    # Versão 1 (TF)
    doc_queries_tokens, query_txt = utils.get_sample_documents(conteudo_tokens, lista_documentos) # 50 documentos aleatórios
    queries_tokens = utils.unify_queries_weights(doc_queries_tokens, term_queries) # vetor com todas as 60 consultas
    queries_txt = utils.unify_queries_txt(query_txt, term_queries)

    # ------- Fazer o ranqueamento de cada consulta (retorna 30 mais similares) ------- #
    # Modelos para as diferentes similaridades
    modelos = ["BM1", "BM11", "BM15", "BM25"]
    results = []
    avg_dl = utils.average_doclen(conteudo_tokens)
    print("Gerando ranks e escrevendo arquivos...\n")
    for model in modelos:
        nome_arquivo_num = f"resultados_numericos_{model}.txt"
        nome_arquivo_txt = f"resultados_textuais_{model}.txt"
        for i, query in enumerate(queries_tokens):
            results.append(ranking.ranqueamento_prob(query, conteudo_tokens, ni_map, N, avg_dl, func_sim=model, K=1.5, b=0.75))
            # ------- Gerar os arquivos de saída ------- #
            writer.write_numeric_file(i + 1, results[i], nome_arquivo_num, "results/atv_2")
            writer.write_textual_file(i, queries_txt, results[i], lista_documentos, nome_arquivo_txt, "results/atv_2")


if __name__ == "__main__":
    main()