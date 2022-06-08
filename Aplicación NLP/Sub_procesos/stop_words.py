from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

ps = PorterStemmer() 

oracion = "un dia tiene 24 horas y una semana son cerca de 420 horas"

#llamamos las stopwords y elegimos el idioma

stopwords = set(stopwords.words('spanish'))
word_tokens = word_tokenize(oracion)

oracion_filtrada = [w for w in word_tokens if not w in stopwords]
oracion_filtrada = []

for w in word_tokens:
    if w not in stopwords:
        oracion_filtrada.append(w)

#print(oracion_filtrada)

#Imprimir en columnas

for w in oracion_filtrada:
    print(ps.stem(w))