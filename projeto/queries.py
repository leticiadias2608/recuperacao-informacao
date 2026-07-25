import pandas as pd
import json
import re
import os

# ==========================================
# CONFIGURAÇÕES
# ==========================================
caminho_fashion_processado = 'dataset_amostra_500/fashion/fashion_processado.csv'
caminho_fashion_styles = 'fashion_styles/styles.csv'
caminho_flickr_processado = 'dataset_amostra_500/flickr/flickr_processado.csv'

pasta_fashion = 'dataset_amostra_500/fashion'
pasta_flickr = 'dataset_amostra_500/flickr'

# ==========================================
# 1. QUERIES DE TESTE - FASHION (30 queries)
# ==========================================
# Divisão de 15 genéricas e 15 específicas (com nome de marca ou modelo na amostra)
# Cada par de consultas representa uma categoria do styles.csv
FASHION_QUERIES = [
    {"texto": "Basic t-shirt for everyday wear", "tipo": "generica",
     "categoria_esperada": "Tshirts", "atributos_esperados": []},
    {"texto": "Grey striped t-shirt by Nike", "tipo": "especifica",
     "categoria_esperada": "Tshirts", "atributos_esperados": ["Nike", "Grey", "Striped"]},
 
    {"texto": "Formal shirt to wear at work", "tipo": "generica",
     "categoria_esperada": "Shirts", "atributos_esperados": []},
    {"texto": "White striped shirt by John Miller", "tipo": "especifica",
     "categoria_esperada": "Shirts", "atributos_esperados": ["John Miller", "White", "Striped"]},
 
    {"texto": "Comfortable shoes for walking a lot", "tipo": "generica",
     "categoria_esperada": "Casual Shoes", "atributos_esperados": []},
    {"texto": "Brown Clarks Rocco Fuse casual shoes", "tipo": "especifica",
     "categoria_esperada": "Casual Shoes", "atributos_esperados": ["Clarks", "Rocco", "Brown"]},
 
    {"texto": "Wrist accessory to tell time", "tipo": "generica",
     "categoria_esperada": "Watches", "atributos_esperados": []},
    {"texto": "Black dial watch by Fastrack for women", "tipo": "especifica",
     "categoria_esperada": "Watches", "atributos_esperados": ["Fastrack", "Black", "Women"]},
 
    {"texto": "Accessory to protect your eyes from strong sun", "tipo": "generica",
     "categoria_esperada": "Sunglasses", "atributos_esperados": []},
    {"texto": "Aviator sunglasses by Mayhem", "tipo": "especifica",
     "categoria_esperada": "Sunglasses", "atributos_esperados": ["Mayhem", "Aviator"]},
 
    {"texto": "Light top to wear in summer", "tipo": "generica",
     "categoria_esperada": "Tops", "atributos_esperados": []},
    {"texto": "White floral top by Forever New", "tipo": "especifica",
     "categoria_esperada": "Tops", "atributos_esperados": ["Forever New", "White", "Floral"]},
 
    {"texto": "Bag to carry everyday belongings", "tipo": "generica",
     "categoria_esperada": "Handbags", "atributos_esperados": []},
    {"texto": "Glossy red handbag by Kiara", "tipo": "especifica",
     "categoria_esperada": "Handbags", "atributos_esperados": ["Kiara", "Red", "Glossy"]},
 
    {"texto": "Shoes for running or physical activity", "tipo": "generica",
     "categoria_esperada": "Sports Shoes", "atributos_esperados": []},
    {"texto": "Black Puma Faas 800 running shoes", "tipo": "especifica",
     "categoria_esperada": "Sports Shoes", "atributos_esperados": ["Puma", "Faas", "Black"]},
 
    {"texto": "Elegant shoes for a formal occasion", "tipo": "generica",
     "categoria_esperada": "Heels", "atributos_esperados": []},
    {"texto": "Classy silver heels by Carlton London", "tipo": "especifica",
     "categoria_esperada": "Heels", "atributos_esperados": ["Carlton London", "Silver"]},
 
    {"texto": "Traditional Indian outfit for women", "tipo": "generica",
     "categoria_esperada": "Kurtas", "atributos_esperados": []},
    {"texto": "Purple printed kurta by Urban Yoga", "tipo": "especifica",
     "categoria_esperada": "Kurtas", "atributos_esperados": ["Urban Yoga", "Purple", "Printed"]},
 
    {"texto": "Accessory to hold money and cards", "tipo": "generica",
     "categoria_esperada": "Wallets", "atributos_esperados": []},
    {"texto": "White Ferrari wallet by Puma", "tipo": "especifica",
     "categoria_esperada": "Wallets", "atributos_esperados": ["Puma", "Ferrari", "White"]},
 
    {"texto": "Accessory to hold up your pants", "tipo": "generica",
     "categoria_esperada": "Belts", "atributos_esperados": []},
    {"texto": "Brown leather belt by New Hide", "tipo": "especifica",
     "categoria_esperada": "Belts", "atributos_esperados": ["New Hide", "Brown"]},
 
    {"texto": "Sandals to wear at the beach or at home", "tipo": "generica",
     "categoria_esperada": "Flip Flops", "atributos_esperados": []},
    {"texto": "Nike Snapper flip flops for women", "tipo": "especifica",
     "categoria_esperada": "Flip Flops", "atributos_esperados": ["Nike", "Snapper", "Women"]},
 
    {"texto": "Backpack for work or college", "tipo": "generica",
     "categoria_esperada": "Backpacks", "atributos_esperados": []},
    {"texto": "Brown American Tourister Wanderer backpack", "tipo": "especifica",
     "categoria_esperada": "Backpacks", "atributos_esperados": ["American Tourister", "Wanderer", "Brown"]},
 
    {"texto": "Open, lightweight shoes for hot days", "tipo": "generica",
     "categoria_esperada": "Sandals", "atributos_esperados": []},
    {"texto": "Brown leather sandals by Enroute", "tipo": "especifica",
     "categoria_esperada": "Sandals", "atributos_esperados": ["Enroute", "Brown"]}
]

# ==========================================
# 2. QUERIES DE TESTE - FLICKR8K (30 queries)
# ==========================================
# Divisão de 15 genéricas e 15 específicas (usa vocabulário das legendas)
# Cada par de consultas representa uma categoria semântica (usando palavras chave)
FLICKR_QUERIES = [
    {"texto": "A pet enjoying a moment of leisure", "tipo": "generica",
     "categoria_esperada": "dog", "atributos_esperados": []},
    {"texto": "Black and white dog running through shallow water", "tipo": "especifica",
     "categoria_esperada": "dog", "atributos_esperados": ["dog", "water"]},
 
    {"texto": "Winter activity outdoors", "tipo": "generica",
     "categoria_esperada": "snow", "atributos_esperados": []},
    {"texto": "Lone skier skiing through the snow", "tipo": "especifica",
     "categoria_esperada": "snow", "atributos_esperados": ["skier", "snow"]},
 
    {"texto": "Fun time near the water", "tipo": "generica",
     "categoria_esperada": "water_beach", "atributos_esperados": []},
    {"texto": "Dog running along a beach near the waves", "tipo": "especifica",
     "categoria_esperada": "water_beach", "atributos_esperados": ["dog", "beach"]},
 
    {"texto": "Children having fun", "tipo": "generica",
     "categoria_esperada": "child", "atributos_esperados": []},
    {"texto": "Child wearing big green fist gloves", "tipo": "especifica",
     "categoria_esperada": "child", "atributos_esperados": ["child", "gloves"]},
 
    {"texto": "A person alone in a contemplative moment", "tipo": "generica",
     "categoria_esperada": "man", "atributos_esperados": []},
    {"texto": "Man in a black jacket with a black beard", "tipo": "especifica",
     "categoria_esperada": "man", "atributos_esperados": ["man", "jacket", "beard"]},
 
    {"texto": "A woman doing an outdoor activity", "tipo": "generica",
     "categoria_esperada": "woman", "atributos_esperados": []},
    {"texto": "Female rugby player running with a ball during a match", "tipo": "especifica",
     "categoria_esperada": "woman", "atributos_esperados": ["female", "rugby"]},
 
    {"texto": "A two-wheeled ride outdoors", "tipo": "generica",
     "categoria_esperada": "bicycle", "atributos_esperados": []},
    {"texto": "Biker waiting as his friend bikes ahead of him", "tipo": "especifica",
     "categoria_esperada": "bicycle", "atributos_esperados": ["biker", "bikes"]},
 
    {"texto": "Intense physical exercise", "tipo": "generica",
     "categoria_esperada": "running", "atributos_esperados": []},
    {"texto": "Two dogs running together on a path covered in leaves", "tipo": "especifica",
     "categoria_esperada": "running", "atributos_esperados": ["dogs", "leaves"]},
 
    {"texto": "Adventure on rocky terrain", "tipo": "generica",
     "categoria_esperada": "mountain_rock", "atributos_esperados": []},
    {"texto": "Climber hanging by a rope between the rocks", "tipo": "especifica",
     "categoria_esperada": "mountain_rock", "atributos_esperados": ["climber", "rope"]},
 
    {"texto": "Playing a sport with a ball", "tipo": "generica",
     "categoria_esperada": "ball_sport", "atributos_esperados": []},
    {"texto": "Small girl chasing a pink ball on a gravel driveway", "tipo": "especifica",
     "categoria_esperada": "ball_sport", "atributos_esperados": ["girl", "gravel"]},
 
    {"texto": "Several people gathered in one place", "tipo": "generica",
     "categoria_esperada": "group_people", "atributos_esperados": []},
    {"texto": "Group of people lounging at a beach", "tipo": "especifica",
     "categoria_esperada": "group_people", "atributos_esperados": ["group", "beach"]},
 
    {"texto": "Open green area for outdoor activities", "tipo": "generica",
     "categoria_esperada": "grass_field", "atributos_esperados": []},
    {"texto": "Bearded man in glasses and a Hawaiian shirt sitting on the grass", "tipo": "especifica",
     "categoria_esperada": "grass_field", "atributos_esperados": ["beard", "hawaiian"]},
 
    {"texto": "A jump in mid-air", "tipo": "generica",
     "categoria_esperada": "jumping", "atributos_esperados": []},
    {"texto": "Brown dog jumping over a pole", "tipo": "especifica",
     "categoria_esperada": "jumping", "atributos_esperados": ["dog", "pole"]},
 
    {"texto": "A piece of clothing in a bright, eye-catching color", "tipo": "generica",
     "categoria_esperada": "red_clothing", "atributos_esperados": []},
    {"texto": "Girl in a red shirt taking a picture with her phone", "tipo": "especifica",
     "categoria_esperada": "red_clothing", "atributos_esperados": ["red shirt", "phone"]},
 
    {"texto": "A moment of playful fun", "tipo": "generica",
     "categoria_esperada": "playing", "atributos_esperados": []},
    {"texto": "Four children playing in the water", "tipo": "especifica",
     "categoria_esperada": "playing", "atributos_esperados": ["children", "water"]}
]

# Regras usadas para definir com quais palavras chave a categoria se relaciona
# e quais imagens são relevantes para cada categoria
REGRAS_CATEGORIA_FLICKR = {
    'dog': r'\bdog[s]?\b', # dog | dogs
    'snow': r'\bsnow\b', # snow
    'water_beach': r'\b(beach|ocean|lake|pool|swim(s|ming)?|water)\b', # beach | ocean | lake | pool | swim | swims | swimming | water
    'child': r'\b(child|children|kid|boys?|girls?|baby)\b', # child | children | kid | boy | boys | girl | girls | baby
    'man': r'\b(man|men)\b', # man | men
    'woman': r'\b(woman|women)\b', # woman | women
    'bicycle': r'\b(?:bik(?:e|es|er|ers|ing)|bicycle[s]?)\b', # bike | bikes | biker | bikers | biking | bicycle | bicycles
    'running': r'\brun(s|ning)?\b', # run | runs | running
    'mountain_rock': r'\b(mountains?|rocks?|cliffs?)\b', # mountain | rock | rocks | cliff | cliffs
    'ball_sport': r'\b(balls?|soccer|football|basketball)\b', # ball | balls | soccer | football | basketball
    'group_people': r'\b(group|people|crowd)\b', # group | people | crowd
    'grass_field': r'\b(grass|field|park)\b', # grass | field | park
    'jumping': r'\bjump(s|ing)?\b', # jump | jumps | jumping
    'red_clothing': r'\bred\b', # red
    'playing': r'\bplay(s|ing)?\b', # play | plays | playing
}

# ==========================================
# 3. SALVAR AS QUERIES EM JSON
# ==========================================
os.makedirs(pasta_fashion, exist_ok=True)
os.makedirs(pasta_flickr, exist_ok=True)

with open(os.path.join(pasta_fashion, 'fashion_queries.json'), 'w', encoding='utf-8') as f:
    json.dump(FASHION_QUERIES, f, ensure_ascii=False, indent=2)

with open(os.path.join(pasta_flickr, 'flickr_queries.json'), 'w', encoding='utf-8') as f:
    json.dump(FLICKR_QUERIES, f, ensure_ascii=False, indent=2)

print(f"{len(FASHION_QUERIES)} queries do Fashion em fashion_queries.json salvas")
print(f"{len(FLICKR_QUERIES)} queries do Flickr em flickr_queries.json salvas")

# ==========================================
# 4. DADOS COMPLETOS - FASHION
# (Fashion já tem as categorias atributos no dataset - styles.csv)
# ==========================================
fashion = pd.read_csv(caminho_fashion_processado) # caminho_imagem e texto_busca
styles = pd.read_csv(caminho_fashion_styles, on_bad_lines='skip') # transforma id para int

fashion['id'] = fashion['caminho_imagem'].str.extract(r'(\d+)\.jpg').astype(int) # pega apenas o string dos números sem o .jpg
styles['id'] = styles['id'].astype(int)
fashion_merge = fashion.merge(
    styles[['id', 'articleType', 'productDisplayName', 'baseColour', 'gender']], 
    on='id', how='left'
)

dados_fashion = fashion_merge[['caminho_imagem', 'articleType', 'productDisplayName', 'baseColour', 'gender']]
dados_fashion.to_csv(os.path.join(pasta_fashion, 'fashion_dados.csv'), index=False)
print(f"Salvo: dados de {len(dados_fashion)} imagens em fashion_dados.csv")

# ==========================================
# 5. dados COMPLETO - FLICKR
# (as categorias são definidas por palavras chave)
# ==========================================
flickr = pd.read_csv(caminho_flickr_processado)

linhas_dados_flickr = []
for _, row in flickr.iterrows():
    legenda = str(row['texto_busca']).lower()
    categorias_encontradas = []

    for nome_categoria, padrao in REGRAS_CATEGORIA_FLICKR.items():
        if re.search(padrao, legenda): # se aparece em alguma regra, coloca como categoria encontrada
            categorias_encontradas.append(nome_categoria)

    # no flickr, uma imagem pode ter mais de uma categoria (ex: dog playing with a kid)
    linhas_dados_flickr.append({
        'caminho_imagem': row['caminho_imagem'],
        'legenda': row['texto_busca'],
        'categorias': ';'.join(categorias_encontradas)  # pode ter mais de uma, ou nenhuma
    })

dados_flickr = pd.DataFrame(linhas_dados_flickr)
dados_flickr.to_csv(os.path.join(pasta_flickr, 'flickr_dados.csv'), index=False)
print(f"Salvo: dados de {len(dados_flickr)} imagens em flickr_dados.csv")

# ==========================================
# 6. VERIFICAÇÃO DE SANIDADE
# Confere se toda categoria_esperada e atributos_esperados nas queries 
# realmente tem pelo menos uma imagem correspondente no dados.
# ==========================================
print("\n--- Verificação: toda query tem pelo menos 1 imagem relevante? ---")

problemas = []

for q in FASHION_QUERIES:
    if q['tipo'] == 'generica':
        qtde = (dados_fashion['articleType'] == q['categoria_esperada']).sum()
    else:
        qtde = 0
        for _, linha in dados_fashion.iterrows():
            texto_produto = f"{linha['productDisplayName']} {linha['baseColour']} {linha['gender']}".lower()
            bate_tudo = True
            for atributo in q['atributos_esperados']:
                if atributo.lower() not in texto_produto:
                    bate_tudo = False
                    break
            if bate_tudo:
                qtde += 1
    if qtde == 0:
        problemas.append(f"[Fashion/{q['tipo']}] '{q['texto']}' sem nenhuma imagem correspondente")
 
for q in FLICKR_QUERIES:
    if q['tipo'] == 'generica':
        qtde = 0
        for lista_categorias in dados_flickr['categorias']:
            if q['categoria_esperada'] in str(lista_categorias).split(';'):
                qtde += 1
    else:
        qtde = 0
        for legenda in dados_flickr['legenda']:
            legenda_lower = str(legenda).lower()
            bate_tudo = True
            for atributo in q['atributos_esperados']:
                if atributo.lower() not in legenda_lower:
                    bate_tudo = False
                    break
            if bate_tudo:
                qtde += 1
    if qtde == 0:
        problemas.append(f"[Flickr/{q['tipo']}] '{q['texto']}' sem nenhuma imagem correspondente")
 
if problemas:
    for p in problemas:
        print("PROBLEMA:", p)
else:
    print("OK: todas as 60 queries têm pelo menos 1 imagem relevante com o novo critério.")