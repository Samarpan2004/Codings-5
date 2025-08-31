# textrank_summarizer.py
# Run: python textrank_summarizer.py input.txt

import sys, nltk, numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer

def summarize(text, n=3):
    sents = sent_tokenize(text)
    if len(sents)<=n: return text
    vec = TfidfVectorizer(stop_words='english').fit_transform(sents)
    sim = (vec * vec.T).toarray()
    G = nx.from_numpy_array(sim)
    scores = nx.pagerank_numpy(G)
    ranked = sorted(((scores[i],s) for i,s in enumerate(sents)), reverse=True)
    summary = " ".join([s for _,s in ranked[:n]])
    return summary

if __name__=="__main__":
    fn = sys.argv[1] if len(sys.argv)>1 else None
    if not fn:
        print("Usage: python textrank_summarizer.py input.txt"); sys.exit(1)
    text = open(fn,encoding='utf8').read()
    print("SUMMARY:\\n", summarize(text, n=5))
