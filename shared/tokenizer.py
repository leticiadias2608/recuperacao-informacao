###------------RESPONSÁVEL POR TOKENIZAR OS DOCUMENTOS------------###
import re


### TOKENIZAÇAO ###
def tokenize(lista_documentos):
    conteudo_tokens = []

    for document in lista_documentos:
        conteudo = document['content']
        
        # remove caracteres especiais
        conteudo = re.sub(r"[^\w\s']", ' ', conteudo)
        
        palavras = conteudo.split()
        
        # remove palavras que começam com número ou $
        palavras = [p for p in palavras if not re.match(r'^[$\d]', p)]
        # remove 's do final das palavras
        palavras = [re.sub(r"'s$", "", p) for p in palavras]
        conteudo_tokens.append(palavras)

    return conteudo_tokens

### ---------------------- ATIVIDADE 4 ------------------------ ###
import nltk
from nltk.corpus import stopwords as sw
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

def remove_stop_words(lista_documentos):
    conteudo_stopwords = []
    nltk.download('stopwords')
    stop_words = set(sw.words('english'))

    for document in lista_documentos:
        palavras = []
        for palavra in document:
            if palavra not in stop_words:
                palavras.append(palavra)
        conteudo_stopwords.append(palavras)

    return conteudo_stopwords

def stemming(lista_documentos): 
    conteudo_stemming = []
    stemmer = SnowballStemmer('english')
    for document in lista_documentos:
        palavras = []
        for palavra in document:
            palavra_stem = stemmer.stem(palavra)
            palavras.append(palavra_stem)
        conteudo_stemming.append(palavras)
    
    return conteudo_stemming