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



def read_and_clean(doc):
	article = []
	article_tokenized = []
	with open(doc, 'r') as INFILE:
		data = INFILE.readlines()
		article = data[0].split('. ')
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
		INFILE.close()
		
	return article, article_tokenized

def similarity(s1, s2, dic):
	#dic = list(set(s1 + s2))
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
	#dic = list(dic.update(w) for w in article_tokenized)
	#dic = list(dic)
	for i in range(len(article_tokenized)):
		for j in range(i):
				matrix[i][j] = matrix[j][i] = similarity(article_tokenized[i], article_tokenized[j], dic)
	return matrix


def summarize(doc, length=5):
	article, tokens = read_and_clean(doc)
	matrix = cosine_similarity_matrix(tokens)
	# Step 3 - Rank sentences in similarity matrix
	similarity_graph = nx.from_numpy_array(matrix)
	scores = nx.pagerank(similarity_graph)

	# Step 4 - Sort the rank and pick top sentences
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
	#print("Summarize Text: \n", ". ".join(summary))
	print("Summarize Text: \n", '. '.join(summary))



if __name__ == '__main__':
	main()