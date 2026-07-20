import pandas as pd
import os
import shutil

# ==========================================
# CONFIGURAÇÕES INICIAIS E PASTAS
# ==========================================
caminho_fashion_csv = 'Fashion_Product_Images_(Small)/styles.csv'
caminho_fashion_img = 'Fashion_Product_Images_(Small)/images'
caminho_flickr_txt = 'Flickr8k_text/Flickr8k.token.txt'

# CORREÇÃO: as imagens do Flickr8k estão direto dentro de 'Flickr8k_Dataset',
# sem subpasta 'images'. O caminho antigo ('Flickr8k_Dataset/images')
# nunca existia, então os.path.exists() falhava para todas as imagens
# e tanto a cópia quanto o CSV ficavam vazios.
caminho_flickr_img = 'Flickr8k_Dataset'

# Pastas de destino (AGORA SEPARADAS)
pasta_amostra = 'dataset_amostra_500'
pasta_fashion_base = os.path.join(pasta_amostra, 'fashion')
pasta_fashion_img_dest = os.path.join(pasta_fashion_base, 'images')
pasta_flickr_base = os.path.join(pasta_amostra, 'flickr')
pasta_flickr_img_dest = os.path.join(pasta_flickr_base, 'images')

# Cria as pastas de destino
os.makedirs(pasta_fashion_img_dest, exist_ok=True)
os.makedirs(pasta_flickr_img_dest, exist_ok=True)

# ==========================================
# 1. PROCESSAMENTO DO FASHION DATASET
# ==========================================
print("Processando e amostrando o Fashion Dataset...")
df_fashion = pd.read_csv(caminho_fashion_csv, on_bad_lines='skip')
df_fashion_amostra = df_fashion.sample(n=500, random_state=42)

linhas_fashion_final = []
fashion_encontradas = 0
fashion_faltando = 0

for _, row in df_fashion_amostra.iterrows():
    img_nome = str(row['id']) + '.jpg'
    origem = os.path.join(caminho_fashion_img, img_nome)
    destino = os.path.join(pasta_fashion_img_dest, img_nome)

    if os.path.exists(origem):
        shutil.copy2(origem, destino)
        descricao = f"{row.get('gender', '')} {row.get('masterCategory', '')} {row.get('articleType', '')} {row.get('baseColour', '')} {row.get('usage', '')} - {row.get('productDisplayName', '')}"

        linhas_fashion_final.append({
            'caminho_imagem': f"images/{img_nome}",
            'texto_busca': descricao.strip().replace("nan ", "")
        })
        fashion_encontradas += 1
    else:
        fashion_faltando += 1

pd.DataFrame(linhas_fashion_final).to_csv(os.path.join(pasta_fashion_base, 'fashion_processado.csv'), index=False)
print(f"Fashion: {fashion_encontradas} imagens copiadas, {fashion_faltando} não encontradas no disco.")

# ==========================================
# 2. PROCESSAMENTO DO FLICKR8K
# ==========================================
print("\nProcessando e amostrando o Flickr8k Dataset...")
dados_flickr = []
with open(caminho_flickr_txt, 'r', encoding='utf-8') as f:
    for linha in f:
        partes = linha.strip().split('\t')
        if len(partes) == 2:
            img_nome_completo = partes[0].split('#')[0]
            texto = partes[1]
            dados_flickr.append({'image': img_nome_completo, 'caption': texto})

df_flickr = pd.DataFrame(dados_flickr)
df_flickr_unico = df_flickr.drop_duplicates(subset=['image'])
df_flickr_amostra = df_flickr_unico.sample(n=500, random_state=42)

linhas_flickr_final = []
flickr_encontradas = 0
flickr_faltando = 0

for _, row in df_flickr_amostra.iterrows():
    img_nome = row['image']
    origem = os.path.join(caminho_flickr_img, img_nome)
    destino = os.path.join(pasta_flickr_img_dest, img_nome)

    if os.path.exists(origem):
        shutil.copy2(origem, destino)

        linhas_flickr_final.append({
            'caminho_imagem': f"images/{img_nome}",
            'texto_busca': row['caption']
        })
        flickr_encontradas += 1
    else:
        flickr_faltando += 1

pd.DataFrame(linhas_flickr_final).to_csv(os.path.join(pasta_flickr_base, 'flickr_processado.csv'), index=False)
print(f"Flickr8k: {flickr_encontradas} imagens copiadas, {flickr_faltando} não encontradas no disco.")

print("\nProcesso concluído com sucesso!")
print("As imagens e os CSVs estão devidamente separados dentro de 'dataset_amostra_500'.")