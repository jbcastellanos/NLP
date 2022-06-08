from string import punctuation
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

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

def remove_punctuation(dirty_string):
	for word in non_words:
		dirty_string = dirty_string.replace(word, '')
	return dirty_string

def process_file(file_name):
	file_content = open(file_name, "r").read()
	# All to lower case
	file_content = file_content.lower()
	# Remove punctuation and spanish stopwords
	file_content = remove_punctuation(file_content)
	file_content = remove_stop_words(file_content)
	return file_content 

nlp_article = process_file("D:\Proyectos\Trabajo UGC\Aplicación NLP\Sub_procesos\original.txt")
sentiment_analysis_article = process_file("D:\Proyectos\Trabajo UGC\Aplicación NLP\Sub_procesos\comparativa.txt")

print(type(nlp_article))
print(type(sentiment_analysis_article))
#TF-IDF
vectorizer = TfidfVectorizer ()
X = vectorizer.fit_transform([nlp_article,sentiment_analysis_article])
#X = count.fit_transform([nlp_article,sentiment_analysis_article])
similarity_matrix = cosine_similarity(X,X)

print('----------------------------------')
print('Similitud:')
print('----------------------------------')
print(similarity_matrix)


