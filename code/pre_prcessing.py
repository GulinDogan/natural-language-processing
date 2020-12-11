import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import string
import re
import unidecode
import nltk

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 

from cont_dict import dic 
from abbr_dict import abbr_dict


def replace_unicode(text):
    unaccented_string = unidecode.unidecode(text)  
    return unaccented_string

def remove_urls_and_userName(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+|(rt)*\s?@[a-z]+[0-9]*_?[a-z]*|#')
    return url_pattern.sub(r'', text)

PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

contractions_dict = dic
contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

def expand_contractions(text, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, text)

def remove_number(text):
    return re.sub(r'[0-9]+', '', text)


# 1. Init Lemmatizer
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def lematize(text):
    # 3. Lemmatize a Sentence with the appropriate POS tag
    text = ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(text)])
    return text
