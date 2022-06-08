from nltk.stem import WordNetLemmatizer

# para lematizar NLKT solo funciona para inglés si se desea en español hay que utilizar otros frameworks

lematizar = WordNetLemmatizer()
print(lematizar.lemmatize("walked"))