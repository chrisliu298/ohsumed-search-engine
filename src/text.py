import re
import string

import nltk
from nltk.corpus import stopwords
from Stemmer import Stemmer

nltk.download("stopwords")

stemmer = Stemmer("english")
punctuation = re.compile("[%s]" % re.escape(string.punctuation))
stop_words = set(stopwords.words("english"))


def split_text(text):
    return text.split()


def lowercase_filter(tokens):
    return [token.lower() for token in tokens]


def stem_filter(tokens):
    return stemmer.stemWords(tokens)


def punctuation_filter(tokens):
    return [punctuation.sub("", token) for token in tokens]


def stopword_filter(tokens):
    return [token for token in tokens if token not in stop_words]


def tokenize(text):
    tokens = split_text(text)
    tokens = lowercase_filter(tokens)
    tokens = punctuation_filter(tokens)
    tokens = stopword_filter(tokens)
    tokens = stem_filter(tokens)
    return [token for token in tokens if token]
