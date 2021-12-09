import pandas as pd

dataframe = pd.read_csv('E:/Dev/Automated-Essay--Scoring/mysite/grader/utils/essays_and_scores.csv', encoding = 'latin-1')
data = dataframe[['essay_set','essay']].copy()

import spacy
spacy.load("en_core_web_sm")
from spacy.lang.en import English
parser = English()
def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
from nltk.stem.wordnet import WordNetLemmatizer
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    tokens = list(filter(lambda x: x!='SCREEN_NAME', tokens))
    return tokens

data["processed_essay"] = data["essay"].apply(lambda x: prepare_text_for_lda(x))

text_data = []
for p_s in data["processed_essay"].values.tolist():
    p_s = list(filter(lambda x: x!='SCREEN_NAME', p_s))
    text_data.append(p_s)

from gensim import corpora
dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]
import gensim
NUM_TOPICS = 8  
ldamodel = gensim.models.LdaModel.load('E:/Dev/Automated-Essay--Scoring/mysite/grader/utils/model8.model')
def check_topic(essay, ldamodel=ldamodel, dict=dictionary, corpus_=corpus):
    NUM_TOPICS = 8
    new_doc = prepare_text_for_lda(essay)
    new_doc_bow = dictionary.doc2bow(new_doc)
    print(ldamodel.get_document_topics(new_doc_bow))
    return max(ldamodel.get_document_topics(new_doc_bow), key=lambda x: x[1])[0]
