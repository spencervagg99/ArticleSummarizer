from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from docx import Document
import networkx as nx
import numpy as np
import os
import sys
import re

#Reads in file based on extension
def read(doc):
	regex = re.compile(
		r'^(?:http|ftp)s?://' # http:// or https://
		r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
		r'localhost|' #localhost...
		r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
		r'(?::\d+)?' # optional port
		r'(?:/?|[/?]\S+)$', re.IGNORECASE)
	if (re.match(regex, doc)):
		print('Scanning HTML link')
		sys.exit(0)
	else:
		filename, ext = os.path.splitext(doc)
		article = []
		if (ext == '.txt'):
			print('Scanning text file')
			with open(doc, 'r') as INFILE:
				data = INFILE.readlines()
				article = data[0].split('. ')
				INFILE.close()
		elif (ext in ['.docx', '.doc']):
			print('Scanning document')
			doc = Document(doc)
			article = []
			for p in doc.paragraphs:
				sentence = p.text.split('. ')
				for s in sentence:
					article.append(s)
			#print(article)
			#sys.exit(0)
		elif (ext == '.pdf'):
			print('Scanning pdf file')
			sys.exit(0)
		else:
			print('Sorry this is an incompatible file.\n' + 
				'Please submit a pdf, doc, docx, html, or txt extension')
			sys.exit(1)
	return article

#Reads in, cleans and tokenizes document
def read_and_clean(doc):
	article = read(doc)
	article_tokenized = []
	sentences = []

	stop_words = set(stopwords.words('english'))
	article_tokenized = []
	stemmer = PorterStemmer()

	for sentence in article:
		tokens = word_tokenize(sentence)
		tokens = [t for t in tokens if not t in stop_words]
		tokens = [stemmer.stem(t) for t in tokens]
		tokens = [t.lower() for t in tokens]
		article_tokenized.append(tokens)
		
	return article, article_tokenized

def similarity(s1, s2, dic):
	#Create the embedded
	v1 = [0] * len(dic)
	v2 = [0] * len(dic)
	for w in s1:
		v1[dic.index(w)] += 1
	for w in s2:
		v2[dic.index(w)] += 1
	return 1 - cosine_distance(v1, v2)



def cosine_similarity_matrix(article_tokenized):
	matrix = np.zeros((len(article_tokenized), len(article_tokenized)))
	dic = set()
	for w in article_tokenized:
		dic.update(w)
	dic = list(dic)
	for i in range(len(article_tokenized)):
		for j in range(i):
				matrix[i][j] = matrix[j][i] = similarity(article_tokenized[i], article_tokenized[j], dic)
	return matrix


def summarize(doc, length=5):
	article, tokens = read_and_clean(doc)
	matrix = cosine_similarity_matrix(tokens)
	similarity_graph = nx.from_numpy_array(matrix)
	scores = nx.pagerank(similarity_graph)
	ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(article)), reverse=True)    
	print("Indexes of top ranked_sentence order are ", ranked_sentence)    

	summarize = []

	for i in range(length):
		summarize.append(ranked_sentence[i][1])

	return summarize


def main():
	#Check for proper arg length, get filenames, initialize invIndex
	print(len(sys.argv), sys.argv)
	if len(sys.argv) != 2 and len(sys.argv) != 3:
		print("Give arguments: Document to be summarized and optionally length of summary")
		sys.exit()
	print('Document to be summarized: %s' % sys.argv[1])
	document = sys.argv[1]
	if (len(sys.argv) == 3):
		length = int(sys.argv[2])
		summary = summarize(document, length)
	else:
		summary = summarize(document)
	print("Summarize Text: \n", '. '.join(summary))



if __name__ == '__main__':
	main()