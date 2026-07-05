from shared import feature_extractor, reader, tokenizer, utils, weighting, writer, ranking, evaluation, feedback

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

def main():
    
    ##-------------------------------------------------------------------------##
    ## ---------------- Extraindo características das imagens ---------------- ##
    ##-------------------------------------------------------------------------##
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[INFO] Dispositivo: {device}")
    
    # 1. Escaneia dataset
    entries, class_names = feature_extractor.scan_dataset("Corel-1K")

    print(f"[INFO] Dataset: {len(entries)} imagens, {len(class_names)} classes")
    print(f"       Classes: {class_names}")
    
    # 2. Carrega modelo
    model = feature_extractor.load_model(device)
    
    # 3. Extrai features
    features, labels, paths = feature_extractor.extract_features(
        entries, model, device, batch_size=32
    )

    # 4. Salva
    feature_extractor.save_features(features, labels, paths, class_names)

    # lista com todas as imagens vetorizadas (features)
    features_list = reader.read_features_file()

    # dicionário das queries com "query_id", "label", "category"
    queries_dict = reader.read_metadata()

    imgs_list = [] # lista de dicionários do tipo {"category": , "content": } 

    for query in queries_dict:
        img_id = query["query_id"]
        # Dicionário da query com apenas categoria e id do documento
        img = {
            "category": query["category"],
            "content": features_list[img_id]
        }
        imgs_list.append(img)

    ##-----------------------------------------------------------------------------##
    ## ---- Fazer o ranqueamento de cada consulta (retorna 50 mais similares) ---- ##
    ##-----------------------------------------------------------------------------##

    # Queremos fazer um ranqueamento com cada uma das imagens do dataset
    # Assim, cada item da features_list será usada como query
    for i, query in enumerate(features_list):
        results = ranking.ranqueamento_euclid(query, features_list)
        writer.write_numeric_file(i + 1, results, "resultados_img_feat.txt", "results/atv_6")
 
    nome_arquivo = f"./results/atv_6/resultados_img_feat.txt"  
    ranked_lists =  reader.read_ranked_file(nome_arquivo)
    
    precision5 = evaluation.precision_calc(queries_dict, ranked_lists, imgs_list, 5)
    precision10 = evaluation.precision_calc(queries_dict, ranked_lists, imgs_list, 10)
    map = evaluation.map_calc(queries_dict, ranked_lists, imgs_list)

    print("Precision@5: ", precision5)
    print("Precision@10: ", precision10)
    print("MAP: ", map)


if __name__ == "__main__":
    main()