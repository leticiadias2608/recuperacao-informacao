from shared import reader, tokenizer, utils, weighting, writer, ranking

def main():
    # Ler o arquivo e gerar lista de dados brutos
    lista_documentos = reader.read_dataset_file()
    conteudo_tokens = tokenizer.tokenize(lista_documentos) # tokenizar os arquivos 
    stop = tokenizer.remove_stop_words(conteudo_tokens)
    tokenizer.stemming(stop) 


if __name__ == "__main__":
    main()