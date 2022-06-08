from fileinput import filename
from unittest import result
from matplotlib.pyplot import title
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import scipy as sp
from string import punctuation
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

#En primer lugar se asigna un input de busqueda
buscar = "Lionel Andrés Messi Cuccittini (Rosario, 24 de junio de 1987), conocido como Leo Messi, es un futbolista argentino que juega como delantero o centrocampista. Jugador histórico del Fútbol Club Barcelona, al que estuvo ligado veinte años, desde 2021 integra el plantel del Paris Saint-Germain de la Ligue 1 de Francia. Es también internacional con la selección de Argentina, equipo del que es capitán y máximo goleador histórico."

#Luego lo asignamos al motor de busqueda y delimitamos las muestras
result = search(buscar, stop=2, lang="es")

#Se crea un búcle en el se recopile los links que tengan similitudes y se extrae el título y la Url
for r in result:
	print("\n\n\n"+r)
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

	print(context)
            
#Stopwords
language_stopwords = stopwords.words('spanish')
non_words = list(punctuation)

def remove_stop_words(dirty_text):
	cleaned_text = ''
	for word in dirty_text.split():
		if word in language_stopwords or word in non_words:
			continue
		else:
			cleaned_text += word + ' '
	return cleaned_text
	
#Puntuacion
def remove_punctuation(dirty_string):
	for word in non_words:
		dirty_string = dirty_string.replace(word, '')
	return dirty_string

#Procesamiento Info
def process_file(file_name):
	file_content = file_name
	file_content = file_content.lower()
	# Remove punctuation and spanish stopwords
	file_content = remove_punctuation(file_content)
	file_content = remove_stop_words(file_content)
	return file_content 

#Documentos
document1 = process_file(context)
document2 = process_file(buscar)

#Similitud
vectorizer = TfidfVectorizer ()
X = vectorizer.fit_transform([document1,document2])
#X = count.fit_transform([nlp_article,sentiment_analysis_article])
similarity_matrix = cosine_similarity(X,X)

resultado = similarity_matrix[1,0]*100

#Mostrar
print('Similitud:')
print(str(resultado, ) + ' %')