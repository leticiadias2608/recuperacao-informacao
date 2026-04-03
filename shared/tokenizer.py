###------------RESPONSÁVEL POR TOKENIZAR OS DOCUMENTOS------------###
import re

### TOKENIZAÇAO ###
def tokenize(lista_documentos):
    conteudo_tokens = []

    for document in lista_documentos[:2]:
        conteudo = document['content']
        
        # remove caracteres especiais
        conteudo = re.sub(r"[^\w\s']", ' ', conteudo)
        
        palavras = conteudo.split()
        
        # remove palavras que começam com número ou $
        palavras = [p for p in palavras if not re.match(r'^[$\d]', p)]
        # remove 's do final das palavras
        palavras = [re.sub(r"'s$", "", p) for p in palavras]
        conteudo_tokens.append(palavras)

    #print(conteudo_tokens)
    return conteudo_tokens
