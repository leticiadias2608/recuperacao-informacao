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
    {"texto": "Roupa básica para o dia a dia", "categoria_esperada": "Tshirts"}, # genérica
    {"texto": "Camiseta listrada cinza da Nike", "categoria_esperada": "Tshirts"}, # específica

    {"texto": "Camisa social para usar no trabalho", "categoria_esperada": "Shirts"}, # genérica
    {"texto": "Camisa branca listrada da John Miller", "categoria_esperada": "Shirts"}, # específica

    {"texto": "Calçado confortável para caminhar bastante", "categoria_esperada": "Casual Shoes"},
    {"texto": "Tênis casual marrom da Clarks modelo Rocco Fuse", "categoria_esperada": "Casual Shoes"},

    {"texto": "Acessório de pulso para ver as horas", "categoria_esperada": "Watches"},
    {"texto": "Relógio Fastrack com mostrador preto feminino", "categoria_esperada": "Watches"},

    {"texto": "Acessório para proteger os olhos do sol forte", "categoria_esperada": "Sunglasses"},
    {"texto": "Óculos de sol aviador da Mayhem", "categoria_esperada": "Sunglasses"},

    {"texto": "Blusa leve para usar no verão", "categoria_esperada": "Tops"},
    {"texto": "Top floral branco da Forever New", "categoria_esperada": "Tops"},

    {"texto": "Bolsa para carregar pertences no dia a dia", "categoria_esperada": "Handbags"},
    {"texto": "Bolsa vermelha brilhante da Kiara", "categoria_esperada": "Handbags"},

    {"texto": "Calçado para praticar corrida ou atividade física", "categoria_esperada": "Sports Shoes"},
    {"texto": "Tênis Puma Faas 800 preto para corrida", "categoria_esperada": "Sports Shoes"},

    {"texto": "Sapato elegante para uma ocasião formal", "categoria_esperada": "Heels"},
    {"texto": "Salto prateado clássico da Carlton London", "categoria_esperada": "Heels"},

    {"texto": "Roupa tradicional indiana feminina", "categoria_esperada": "Kurtas"},
    {"texto": "Kurta estampada roxa da Urban Yoga", "categoria_esperada": "Kurtas"},

    {"texto": "Acessório para guardar dinheiro e cartões", "categoria_esperada": "Wallets"},
    {"texto": "Carteira branca da Puma com estampa Ferrari", "categoria_esperada": "Wallets"},

    {"texto": "Acessório para segurar a calça na cintura", "categoria_esperada": "Belts"},
    {"texto": "Cinto de couro marrom da New Hide", "categoria_esperada": "Belts"},

    {"texto": "Chinelo para usar na praia ou em casa", "categoria_esperada": "Flip Flops"},
    {"texto": "Chinelo Nike Snapper feminino", "categoria_esperada": "Flip Flops"},

    {"texto": "Mochila para o trabalho ou faculdade", "categoria_esperada": "Backpacks"},
    {"texto": "Mochila American Tourister Wanderer marrom", "categoria_esperada": "Backpacks"},

    {"texto": "Calçado aberto e leve para dias quentes", "categoria_esperada": "Sandals"},
    {"texto": "Sandália de couro marrom da Enroute", "categoria_esperada": "Sandals"},
]

# ==========================================
# 2. QUERIES DE TESTE - FLICKR8K (30 queries)
# ==========================================
# Divisão de 15 genéricas e 15 específicas (usa vocabulário das legendas)
# Cada par de consultas representa uma categoria semântica (usando palavras chave)
FLICKR_QUERIES = [
    {"texto": "Animal de estimação em um momento de lazer", "categoria_esperada": "dog"}, # genérica
    {"texto": "Cachorro preto e branco correndo na água rasa", "categoria_esperada": "dog"}, # específica

    {"texto": "Atividade de inverno ao ar livre", "categoria_esperada": "snow"}, # genérica
    {"texto": "Esquiadora com chapéu verde de flores rosa", "categoria_esperada": "snow"}, # específica

    {"texto": "Diversão perto da água", "categoria_esperada": "water_beach"},
    {"texto": "Cachorro correndo na praia perto das ondas", "categoria_esperada": "water_beach"},

    {"texto": "Crianças se divertindo", "categoria_esperada": "child"},
    {"texto": "Criança usando luvas verdes grandes", "categoria_esperada": "child"},

    {"texto": "Pessoa sozinha em um momento contemplativo", "categoria_esperada": "man"},
    {"texto": "Homem de jaqueta preta e barba preta", "categoria_esperada": "man"},

    {"texto": "Mulher praticando uma atividade ao ar livre", "categoria_esperada": "woman"},
    {"texto": "Jogadora de rúgbi correndo com a bola durante a partida", "categoria_esperada": "woman"},

    {"texto": "Passeio de duas rodas ao ar livre", "categoria_esperada": "bicycle"},
    {"texto": "Ciclista esperando o amigo que também está de bicicleta", "categoria_esperada": "bicycle"},

    {"texto": "Prática de exercício físico intenso", "categoria_esperada": "running"},
    {"texto": "Dois cachorros correndo juntos por um caminho coberto de folhas", "categoria_esperada": "running"},

    {"texto": "Aventura em terreno rochoso", "categoria_esperada": "mountain_rock"},
    {"texto": "Escalador pendurado por uma corda entre as rochas", "categoria_esperada": "mountain_rock"},

    {"texto": "Prática de esportes com bola", "categoria_esperada": "ball_sport"},
    {"texto": "Menina pequena correndo atrás de uma bola rosa no cascalho", "categoria_esperada": "ball_sport"},

    {"texto": "Várias pessoas reunidas em um mesmo local", "categoria_esperada": "group_people"},
    {"texto": "Grupo de pessoas deitado relaxando em uma praia", "categoria_esperada": "group_people"},

    {"texto": "Área verde aberta para atividades ao ar livre", "categoria_esperada": "grass_field"},
    {"texto": "Homem de barba escura sentado na grama de óculos e camisa havaiana", "categoria_esperada": "grass_field"},

    {"texto": "Movimento de salto no ar", "categoria_esperada": "jumping"},
    {"texto": "Cachorro marrom pulando sobre um poste", "categoria_esperada": "jumping"},

    {"texto": "Peça de roupa em uma cor vibrante e chamativa", "categoria_esperada": "red_clothing"},
    {"texto": "Menina de camiseta vermelha tirando foto com o celular", "categoria_esperada": "red_clothing"},

    {"texto": "Momento de brincadeira e diversão", "categoria_esperada": "playing"},
    {"texto": "Quatro crianças brincando na água", "categoria_esperada": "playing"},
]

# Regras usadas para definir com quais palavras chave a categoria se relaciona
# e quais imagens são relevantes para cada categoria
REGRAS_CATEGORIA_FLICKR = {
    'dog': r'\bdog[s]?\b', # dog | dogs
    'snow': r'\bsnow\b', # snow
    'water_beach': r'\b(beach|ocean|lake|pool|swim(s|ming)?|water)\b', # beach | ocean | lake | pool | swim | swims | swimming | water
    'child': r'\b(child|children|kid|boy|girl|baby)\b', # child | children | kid | boy | girl | baby
    'man': r'\b(man|men)\b', # man | men
    'woman': r'\b(woman|women)\b', # woman | women
    'bicycle': r'\b(bike|bicycle|biking)\b', # bike | bicycle | biking
    'running': r'\brun(s|ning)?\b', # run | runs | running
    'mountain_rock': r'\b(mountain|rock|cliff)\b', # mountain | rock | cliff
    'ball_sport': r'\b(ball|soccer|football|basketball)\b', # ball | soccer | football | basketball
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
# 4. GABARITO COMPLETO - FASHION
# (Fashion já tem as categorias no dataset - styles.csv)
# ==========================================
fashion = pd.read_csv(caminho_fashion_processado) # caminho_imagem e texto_busca
styles = pd.read_csv(caminho_fashion_styles, on_bad_lines='skip') # transforma id para int

fashion['id'] = fashion['caminho_imagem'].str.extract(r'(\d+)\.jpg').astype(int) # pega apenas o string dos números sem o .jpg
styles['id'] = styles['id'].astype(int)
fashion_merge = fashion.merge(styles[['id', 'articleType']], on='id', how='left')

gabarito_fashion = fashion_merge[['caminho_imagem', 'articleType']].rename(
    columns={'articleType': 'categoria'}
)
gabarito_fashion.to_csv(os.path.join(pasta_fashion, 'fashion_gabarito.csv'), index=False)
print(f"Salvo: gabarito de {len(gabarito_fashion)} imagens em fashion_gabarito.csv")

# ==========================================
# 5. GABARITO COMPLETO - FLICKR
# (as categorias são definidas por palavras chave)
# ==========================================
flickr = pd.read_csv(caminho_flickr_processado)

linhas_gabarito_flickr = []
for _, row in flickr.iterrows():
    legenda = str(row['texto_busca']).lower()
    categorias_encontradas = []

    for nome_categoria, padrao in REGRAS_CATEGORIA_FLICKR.items():
        if re.search(padrao, legenda): # se aparece em alguma regra, coloca como categoria encontrada
            categorias_encontradas.append(nome_categoria)

    # no flickr, uma imagem pode ter mais de uma categoria (ex: dog playing with a kid)
    linhas_gabarito_flickr.append({
        'caminho_imagem': row['caminho_imagem'],
        'categorias': ';'.join(categorias_encontradas)  # pode ter mais de uma, ou nenhuma
    })

gabarito_flickr = pd.DataFrame(linhas_gabarito_flickr)
gabarito_flickr.to_csv(os.path.join(pasta_flickr, 'flickr_gabarito.csv'), index=False)
print(f"Salvo: gabarito de {len(gabarito_flickr)} imagens em flickr_gabarito.csv")

# ==========================================
# 6. VERIFICAÇÃO DE SANIDADE
# Confere se toda categoria_esperada usada nas queries realmente tem
# pelo menos uma imagem correspondente no gabarito.
# ==========================================
print("\n--- Verificação: categorias das queries existem no gabarito? ---")

problemas = []

for q in FASHION_QUERIES:
    cat = q['categoria_esperada']
    qtde = (gabarito_fashion['categoria'] == cat).sum()
    if qtde == 0:
        problemas.append(f"[Fashion] '{cat}' sem nenhuma imagem correspondente")

for q in FLICKR_QUERIES:
    cat = q['categoria_esperada']
    qtde = 0
    for lista_categorias in gabarito_flickr['categorias']:
        if cat in lista_categorias.split(';'):
            qtde += 1
    if qtde == 0:
        problemas.append(f"[Flickr] '{cat}' sem nenhuma imagem correspondente")

if problemas:
    for p in problemas:
        print("PROBLEMA:", p)
else:
    print("Verificação OK: todas as categorias das queries têm imagens correspondentes.")