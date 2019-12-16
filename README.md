# ArticleSummarizer
This project looks to explore Natural Language Processing through text summarization. I am currently working on building out an extractive summarization model, and I hope to also be able to build an abstractive model.

## Extractive Summarizer
This program takes one argument with an optional second. The first argument is the file path. Currently, the program supports .txt, .docx, and .doc files, with support toward HTML and .pdf files coming soon. The second argument is the number of sentences you wish to be used. If n > total number of sentences, then the program will just return the original article. An example call would be: 

```python extractiveSummarizer.py msft.txt 2```
