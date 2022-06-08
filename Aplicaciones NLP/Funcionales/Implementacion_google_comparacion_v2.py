from unittest import result
from matplotlib.pyplot import title
from matplotlib.style import context
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import scipy as sp
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re, unicodedata

# ---> Funciones de Filtrado

def remove_non_ascii(words):
    """Los carácteres no pertenecientes a ASCI los remueve"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convierte cada palabra en minuscula"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remueve los signos de puntución"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def remove_stopwords(words):
    """Remueve las palabras más comunes"""
    new_words = []
    for word in words:
        if word not in stopwords.words('spanish'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words"""
    stemmer = SnowballStemmer('spanish')
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lematizar palabras en inglés"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    """En normalize se aplica la etapa de filtrado"""
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    words = stem_words(words)
    return words

# ---> Texto Input 

texto_1 = "Solo sé que nada sé"
contexto_1 = texto_1
texto_1 = word_tokenize(texto_1) 
texto_1 = normalize(texto_1)

print(texto_1)

# ---> Texto Google 

result = search(contexto_1, stop=1, lang="es")

for r in result:
	print("\n"+r)
	paginas = requests.get(r)
	soup= BeautifulSoup(paginas.content, "html5lib")
	textos = soup.find_all(["p", "li", "strong"])
	# print(str(res)+'\n\n')
	context=""
	for texto in textos:
		this_context = texto.get_text()
		if this_context.count(' ') > 7:
			context += "\n"
			context += this_context

texto_2 = word_tokenize(context)
texto_2 = normalize(texto_2)

print(texto_2)

# Establecer similitud a través de TF-IDF

vectorizer = TfidfVectorizer ()
X = vectorizer.fit_transform([str(texto_1),str(texto_2)])
similarity_matrix = cosine_similarity(X,X)

#Conversión porcentaje 

resultado = similarity_matrix[1,0] * 100

#Mostrar

print('Similitud:')
print(str(resultado) + ' %')