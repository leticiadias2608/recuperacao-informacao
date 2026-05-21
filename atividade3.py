from shared import reader, tokenizer, utils, weighting, writer, ranking, evaluation

def main():
    # Ler o arquivo e gerar lista de dados brutos
    lista_documentos = reader.read_dataset_file()
    
    # Leitura das consultas
    queries_dict = reader.read_queries_file("data/queries.txt") 
    
    resultados = []

    modelos_atv1 = ["TF", "IDF", "TF-IDF"]
    
    for model in modelos_atv1:
        nome_arquivo = f"./results/atv_1/resultados_numericos_{model}.txt"  
        ranked_lists =  reader.read_rankeds_file(nome_arquivo)
        
        metricas = evaluation.evaluate(queries_dict, ranked_lists, lista_documentos)
        metricas["Modelo"] = model
        resultados.append(metricas)  

        print(f"\n=== {model} ===")
        for nome, valor in metricas.items(): 
            if isinstance(valor, (int, float)): # printa só os valores numéricos
                print(f"{nome}: {valor:.4f}")
    
    modelos_atv2 = ["BM1", "BM11", "BM15", "BM25"]
    for model in modelos_atv2:
        nome_arquivo = f"./results/atv_2/resultados_numericos_{model}.txt"
        ranked_lists = reader.read_rankeds_file(nome_arquivo)

        metricas = evaluation.evaluate(queries_dict, ranked_lists, lista_documentos)
        metricas["Modelo"] = model
        resultados.append(metricas)
        
        print(f"\n=== {model} ===")
        for nome, valor in metricas.items(): 
            if isinstance(valor, (int, float)): # printa só os valores numéricos
                print(f"{nome}: {valor:.4f}")

    writer.write_results_tables(resultados, "tabela_avaliacao_modelos", "results/atv_3")

if __name__ == "__main__":
    main()