import math
import numpy as np

class BM25:
    def __init__(self,corpus,k1=1.5,b=0.75,epsilon=0.25):
        self.corpus = corpus #语料库
        self.corpus_size = 0
        self.doc_len = []
        self.doc_freqs=[]
        self.avgdl = 0
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.df = {}
        num_doc = 0
        for document in self.corpus:
            self.doc_len.append(len(document))
            num_doc+=len(document)
            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word]=0
                frequencies[word]+=1
            self.doc_freqs.append(frequencies)
            for word,freq in frequencies.items():
                try:
                    self.df[word]+=1
                except KeyError:
                    self.df[word]=1
            self.corpus_size+=1
        self.avgdl = num_doc/self.corpus_size
        self.idf = {}
        idf_sum = 0
        negative_idfs = []
        for term,freq in self.df.items():
            self.idf[term] = math.log((self.corpus_size-freq+0.5)/(freq+0.5)) # 计算idf
            idf_sum+=math.log((self.corpus_size-freq+0.5)/(freq+0.5))
            if math.log((self.corpus_size-freq+0.5)/(freq+0.5))<0:
                negative_idfs.append(term)
        self.average_idf = idf_sum/len(self.idf)
        eps = self.epsilon*self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps
    def get_scores(self,query):
        scores = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for  doc in self.doc_freqs])
            scores+=(self.idf.get(q) or 0) * (q_freq*(self.k1+1)/
                                              (q_freq+self.k1*(1-self.b+self.b*doc_len/self.avgdl)))
        return list(scores)
                    