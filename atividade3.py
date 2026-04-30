from shared import reader, tokenizer, utils, weighting, writer, ranking, evaluation

def main():
    # Ler o arquivo e gerar lista de dados brutos
    lista_documentos = reader.read_dataset_file()
    
    # Leitura das consultas
    queries_dict = reader.read_queries_file() 
    
    modelos_atv1 = ["tf", "idf", "tf_idf"]
    
    for model in modelos_atv1:
        nome_arquivo = f"./results/atv_1/resultados_numericos_{model}.txt"  
        ranked_lists =  reader.read_rankeds_file(nome_arquivo)
        
        metricas = evaluation.evaluate(queries_dict, ranked_lists, lista_documentos)
            
        print(f"\n=== {model} ===")
        for nome, valor in metricas.items():
            print(f"{nome}: {valor:.4f}")
    
    modelos_atv2 = ["BM1", "BM11", "BM15", "BM25"]
    for model in modelos_atv2:
        nome_arquivo = f"./results/atv_2/resultados_numericos_{model}.txt"
        ranked_lists = reader.read_rankeds_file(nome_arquivo)
        metricas = evaluation.evaluate(queries_dict, ranked_lists, lista_documentos)
        print(f"\n=== {model} ===")
        for nome, valor in metricas.items():
            print(f"{nome}: {valor:.4f}")

if __name__ == "__main__":
    main()