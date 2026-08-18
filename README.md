# Recuperação de Informação

Implementação, do zero, dos principais modelos e técnicas de **Recuperação de Informação (RI)** — do clássico espaço vetorial (TF-IDF) e modelos probabilísticos (BM25) até busca multimodal com embeddings de deep learning (CLIP e S-BERT). O repositório reúne cinco atividades progressivas da disciplina de Recuperação de Informação e um projeto final aplicado, que constrói um mecanismo de busca de imagens por texto sobre datasets de moda e fotografias.

> Projeto acadêmico desenvolvido em Python, com foco em implementar os algoritmos manualmente (sem bibliotecas de busca prontas) para consolidar o entendimento teórico dos modelos de RI, complementado por um caso de uso real com modelos pré-treinados de linguagem e visão.

## Visão geral

O projeto está dividido em duas partes:

1. **Atividades 1 a 5** — evolução incremental de um motor de busca textual construído do zero sobre uma base de notícias da BBC (~2.200 documentos), cobrindo:
   - Modelo vetorial (TF, IDF, TF-IDF) com similaridade de cosseno
   - Modelos probabilísticos de ranqueamento (BM1, BM11, BM15, BM25)
   - Avaliação formal dos modelos (Precisão, Revocação, MAP, NDCG)
   - Impacto de pré-processamento (remoção de stopwords e stemming) na qualidade e no desempenho da busca
   - Realimentação de relevância (*relevance feedback*) com o algoritmo de Rocchio

2. **Projeto final** — sistema de **busca multimodal texto→imagem**, comparando um modelo de linguagem (S-BERT, sobre metadados textuais das imagens) com um modelo multimodal (CLIP, buscando diretamente pela imagem), avaliados sobre datasets de moda (Fashion Product Images) e fotos genéricas (Flickr), com consultas genéricas e específicas.

## Estrutura do repositório

```
recuperacao-informacao/
├── atividade1.py          # Modelo vetorial: TF, IDF, TF-IDF + similaridade de cosseno
├── atividade2.py          # Modelos probabilísticos: BM1, BM11, BM15, BM25
├── atividade3.py          # Avaliação dos modelos das atividades 1 e 2
├── atividade4.py          # Efeito de stopwords/stemming na qualidade e tempo de busca
├── atividade5.py          # Realimentação de relevância (algoritmo de Rocchio)
│
├── shared/                 # Módulos reutilizados pelas atividades 1-5
│   ├── reader.py            # Leitura do dataset e dos arquivos de consultas/resultados
│   ├── tokenizer.py          # Tokenização, stopwords e stemming
│   ├── utils.py               # Vocabulário, geração de consultas, cálculo de ni, índice
│   ├── weighting.py            # Vetorização (TF, TF-log, IDF, TF-IDF)
│   ├── ranking.py                # Similaridade de cosseno e funções de ranqueamento probabilístico
│   ├── evaluation.py              # Métricas: Precisão, Revocação, MAP, NDCG
│   ├── feedback.py                 # Algoritmo de Rocchio
│   └── writer.py                    # Geração dos arquivos de resultado, tabelas e gráficos
│
├── data/
│   ├── bbc-news-data.csv    # Dataset de notícias BBC (business, entertainment, politics, sport, tech)
│   ├── queries.txt           # Consultas geradas para as atividades 1-4
│   └── queries_2.txt          # Consultas geradas para a atividade 5
│
└── projeto/                 # Projeto final: busca multimodal de imagens
    ├── main.py                # Orquestra geração de embeddings, buscas e avaliação
    ├── embeddings_clip.py       # Codificação de imagens/texto com CLIP (openai/clip-vit-base-patch32)
    ├── sbert_utils.py             # Codificação de texto com S-BERT (all-MiniLM-L6-v2)
    ├── queries_evaluation.py       # Gabarito de relevância e cálculo de métricas por dataset
    ├── gerar_graficos.py             # Geração de tabelas-resumo e gráficos comparativos
    ├── sample_datasets_script.py      # Amostragem dos datasets originais (500 imagens)
    ├── extrair_exemplos.py             # Extração de exemplos para o relatório
    ├── criar_grid.py                    # Montagem de grids de imagens para visualização
    └── dataset_amostra_500/
        ├── fashion/            # Amostra do Fashion Product Images Dataset + metadados
        └── flickr/              # Amostra de fotos do Flickr8k/Flickr30k + legendas
```

## Atividades

| # | Tema | O que foi implementado |
|---|------|-------------------------|
| 1 | Modelo vetorial | Vetorização TF, IDF e TF-IDF construída manualmente, ranqueamento por similaridade de cosseno, geração automática de consultas por termo e por documento |
| 2 | Modelos probabilísticos | Implementação das funções de ranqueamento BM1, BM11, BM15 e BM25 |
| 3 | Avaliação de modelos | Cálculo de Precisão@N, Revocação, MAP (Mean Average Precision) e NDCG para comparar todos os modelos das atividades 1 e 2 |
| 4 | Pré-processamento | Comparação de 4 variantes (sem pré-processamento, apenas stopwords, apenas stemming, ambos) medindo qualidade dos resultados, tamanho do vocabulário e tempo de execução |
| 5 | Realimentação de relevância | Reescrita iterativa das consultas com o algoritmo de Rocchio, avaliando o ganho de desempenho ao longo de 10 iterações, inclusive sobre documentos ainda não vistos (avaliação residual) |

Cada atividade gera arquivos de resultado (rankings numéricos e textuais) na pasta `results/`, além de tabelas comparativas e, na atividade 5, gráficos de evolução das métricas.

## Projeto final: busca multimodal de imagens

O projeto aplica os conceitos de RI a um cenário mais próximo do estado da arte: buscar imagens a partir de uma consulta em texto livre.

- **S-BERT** (`sentence-transformers/all-MiniLM-L6-v2`) gera embeddings a partir de metadados textuais associados a cada imagem (título, cor, categoria, legenda) e busca por similaridade de cosseno texto-texto.
- **CLIP** (`openai/clip-vit-base-patch32`) codifica as imagens diretamente com seu encoder visual e a consulta com seu encoder de texto, permitindo busca texto-imagem sem depender de metadados.
- As duas abordagens são avaliadas lado a lado em dois datasets (moda e fotografia genérica), com consultas **genéricas** (ex.: "sapato preto") e **específicas** (ex.: "tênis de corrida azul e branco"), usando Precisão@K, Revocação@K e MAP@K.

O objetivo é comparar, na prática, um modelo de linguagem que depende de boas descrições textuais com um modelo multimodal que "enxerga" a imagem — e entender em que cenários cada um se sai melhor.

## Tecnologias utilizadas

- **Python 3**
- **NumPy** — operações vetoriais e cálculo de similaridade
- **Pandas** — manipulação dos datasets e metadados
- **NLTK** — remoção de stopwords e stemming (Porter/Snowball)
- **Matplotlib** — geração de gráficos comparativos
- **PyTorch** + **Transformers (Hugging Face)** — modelo CLIP
- **Sentence-Transformers** — modelo S-BERT

## Como executar

```bash
# Instalar dependências
pip install numpy pandas nltk matplotlib torch transformers sentence-transformers pillow

# Atividades 1-5 (executar a partir da raiz do repositório)
python atividade1.py
python atividade2.py
python atividade3.py
python atividade4.py
python atividade5.py

# Projeto final (executar a partir da pasta projeto/)
cd projeto
python main.py
python gerar_graficos.py
```

Os resultados numéricos e textuais de cada atividade são salvos em `results/atv_N/`, e os do projeto final em `projeto/results/`, junto com as tabelas e gráficos comparativos gerados por `gerar_graficos.py`.

## Datasets

- **BBC News Dataset** — cerca de 2.200 notícias divididas em 5 categorias (negócios, entretenimento, política, esporte e tecnologia), usadas como coleção de teste nas atividades 1-5.
- **Fashion Product Images Dataset** — amostra de 500 imagens de produtos de moda com metadados (categoria, cor, gênero, nome do produto).
- **Flickr (8k/30k)** — amostra de 500 fotos com legendas e categorias, usada como coleção mais heterogênea para testar a generalização dos modelos.

---

Projeto desenvolvido para fins acadêmicos e de portfólio, como parte da disciplina de Recuperação de Informação.
