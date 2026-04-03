###------------RESPONSÁVEL POR LER ARQUIVOS E DADOS SALVOS------------###

### LEITURA DO ARQUIVO (DATASET) ###
def read_dataset_file():
    dataset_path = "data/bbc-news-data.csv" # caminho para o dataset

    with open(dataset_path, 'r', encoding='utf-8') as arquivo:
        linhas = arquivo.readlines()
        
        cabecalho = linhas[0].lower().strip().split('\t')
        #print(cabecalho)
        
        lista_documentos = []
        for linha in linhas[1:]:
            valores = linha.lower().strip().split('\t')
            documento = dict(zip(cabecalho, valores))
            #print(valores)
            lista_documentos.append(documento)

        return lista_documentos
        #print(documento)
        #print(lista_documentos)

