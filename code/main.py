import pandas as pd
import string
import seaborn as sns
from sklearn.model_selection import train_test_split

from prepro import *
from abbr_dict import *
from visualization import *
from ml import ml 

txt = pd.read_csv('C:/Users/GulinDogan/Desktop/NLP/Proje/Data/GOP_REL_ONLY.csv')

sentiment = txt.groupby("sentiment")
print(sentiment.describe().head(20))

sentiment_encodeing = sentimentGrafic(txt["sentiment"])

print("Missing value:\n",txt.isnull().sum())

df = txt[["text"]]
df["text"] = df["text"].astype(str)

# Remove blank rows if any.
if (len(txt['text']) == 0) | (len(txt['sentiment']) == 0) :
    txt.dropna(inplace=True)

print("Orjinal: \n", df["text"].head(20))

df["replace_unicode"] = df["text"].apply(lambda text: replace_unicode(text))
print("\n",df["replace_unicode"].head(20))

df["text_lower"] = df["replace_unicode"].str.lower()
print("\n",df["text_lower"].head(20))

df["remove_urls_and_userName"] = df["text_lower"].apply(lambda text: remove_urls_and_userName(text))
print("\n",df["remove_urls_and_userName"].head(20))

df["replace_abbreviation"] = df["remove_urls_and_userName"].apply(lambda text: expand_contractions(text))
print("\n",df["replace_abbreviation"].head(20))

df["remove_numb"] = df["replace_abbreviation"].apply(lambda text: remove_number(text))
print("\n",df["remove_numb"].head(20))

df["remove_punctuation"] = df["remove_numb"].apply(lambda text: remove_punctuation(text))
print("\n", df["remove_punctuation"].head(20))
# print(type(df["remove_punctuation"]))

df["steamming"] = df["remove_punctuation"].apply(lambda text: lematize(text))
print("\n", df["steamming"].head(20))

print("En çok tekrar edeen kelimeler: \n")

cloud(df["steamming"])

# Splitting the dataset into train and test set
Train_X, Test_X, Train_Y, Test_Y =  train_test_split(df["steamming"], sentiment_encodeing,  test_size = 0.2)

ml(Train_X, Test_X, Train_Y, Test_Y)
    
