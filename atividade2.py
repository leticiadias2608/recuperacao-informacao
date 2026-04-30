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

    # Converte vetor_ni (lista) para dicionário {termo: ni}
    ni_map = {termo: vetor_ni[i] for i, termo in enumerate(vocabulario)}

    # Leitura das consultas
    queries_dict = reader.read_queries_file() 

    ##---------------------------------------------------------------------------------##
    # ------- Fazer o ranqueamento de cada consulta (retorna 30 mais similares) ------- #
    ##---------------------------------------------------------------------------------##
    # Modelos para as diferentes similaridades
    modelos = ["BM1", "BM11", "BM15", "BM25"]
    avg_dl = utils.average_doclen(conteudo_tokens)
    print("Gerando ranks e escrevendo arquivos...\n")
    for model in modelos:
        nome_arquivo_num = f"resultados_numericos_{model}.txt"
        nome_arquivo_txt = f"resultados_textuais_{model}.txt"
        for i, query in enumerate(queries_dict):
            result = ranking.ranqueamento_prob(query["content"], conteudo_tokens, ni_map, N, avg_dl, func_sim=model, K=1.5, b=0.75)
            # ------- Gerar os arquivos de saída ------- #
            writer.write_numeric_file(i + 1, result, nome_arquivo_num, "results/atv_2")
            writer.write_textual_file(i, queries_dict[i]["content"], result, lista_documentos, nome_arquivo_txt, "results/atv_2")


if __name__ == "__main__":
    main()