import json
import numpy as np
from string import punctuation
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import nltk.tokenize as tk
import nltk
nltk.download('punkt')
class preprocessing:
    def __init__(self,fn):
        self.fn = fn
        jf = open("../crawler/urls.txt","r")
        data = json.load(jf)
        self.inclusions = data["inclusion_list"]
        jf.close()
    def load_docs(self):
        jf = open(self.fn, 'r',encoding='utf-8', errors='ignore')
        json_data = json.load(jf)
        jf.close()
        return json_data
    def extract_data(self,json_data):
        paras = []
        sents = []
        labels = []
        links = []
        for k, v in json_data.items():
            ''' Sentences Extraction
            tokens_len = 0
            for p in v:
                tokens = tk.sent_tokenize(p)
                sents.extend(tokens)
                tokens_len += len(tokens)'''
            links.extend(np.repeat(k,len(v)))
            paras.extend(v)
            for url in self.inclusions:
                if k.startswith(url):
                    labels.extend(np.repeat(self.inclusions.index(url), len(v)))
                    break
        return paras, labels, links

    def clean_docs(self,documents):
        cleaned_documents = []
        table = str.maketrans('','',punctuation)
        stop_words = set(stopwords.words("english"))
        for doc in documents:
            tokens = doc.split()
            # remove all punctuations
            tokens = [w.translate(table) for w in tokens]
            # remove remaining tokens that are not alphabetic
            tokens = [word for word in tokens if word.isalpha()]
            tokens = [wd.lower() for wd in tokens]
            # filter out stop words
            tokens = [w for w in tokens if w not in stop_words]
            # filter out short tokens
            tokens = [word for word in tokens if len(word)>1]
            # convert list to string
            cleaned_doc = ' '.join(tokens)
            cleaned_documents.append(cleaned_doc)
        return cleaned_documents
    def encode_labels(self,labels):
        # transform labels into vectors
        numerical_labels = []
        for i in range(len(labels)):
            if i == 0:
                numerical_labels.append(0)
            else:
                if labels[i] == labels[i - 1]:
                    numerical_labels.append(numerical_labels[i - 1])
                else:
                    numerical_labels.append(numerical_labels[i - 1] + 1)
        y = np.array(numerical_labels)
        return y
    def process(self):
        json_data = self.load_docs()
        paras, labels, links = self.extract_data(json_data)
        paras = self.clean_docs(paras)
        return paras, labels, links
    def trim_paras(self,paras):
        """"""
if __name__ == "__main__":
    pt = preprocessing("../crawler/clcjsondata.txt")
    json_data = pt.load_docs()
    docs, labels,links = pt.process()
    print(docs, labels, links)
