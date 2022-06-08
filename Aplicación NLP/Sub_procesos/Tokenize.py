from operator import imod
from tokenize import Token
from matplotlib.pyplot import draw
import nltk
from nltk.corpus import treebank


#Funciones de tokenizar, etiquetas y entidades

sentence = "A las 8 de la madrugada Facebook no se setía bien"
tokens = nltk.word_tokenize(sentence)
#print(tokens)

etiqueta = nltk.pos_tag(tokens)
#print(etiqueta)

entidades = nltk.chunk.ne_chunk(etiqueta)

#print(entidades)

# Ejemplo diagrama de árbol

#t = treebank.parsed_sents('wsj_0001.mrg')[0]
#t.draw(sentence)

#Prueba de corpus de treebank, brown, inauguration

corpus=nltk.corpus.treebank.fileids()
#print(corpus[1])

#print(treebank.words())
print(treebank.tagged_sents(tagset='universal'))