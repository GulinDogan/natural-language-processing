import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS 
from sklearn.preprocessing import LabelEncoder

import itertools

def cloud(text):
    comment_words = '' 
    stopwords = set(STOPWORDS) 

    for val in text: 
        val = str(val) 
        tokens = val.split() 

        for i in range(len(tokens)): 
            tokens[i] = tokens[i].lower() 
        
        comment_words += " ".join(tokens)+" "
    
    wordcloud = WordCloud(width = 700, height = 700, 
                    background_color ='white', 
                    stopwords = stopwords, 
                    min_font_size = 10).generate(comment_words) 
    
    # plot the WordCloud image                        
    plt.figure(figsize = (7,7), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    
    plt.show() 

def sentimentGrafic(data):

 # sentiment: pos. neg 
    Encoder = LabelEncoder()
    sentiment_encodeing = Encoder.fit_transform(data.astype(str))
    print(sentiment_encodeing)
    
    neg = (sentiment_encodeing == 0).sum() 
    #Summing up all the values of column gender with a 
    #condition for male and similary for females
    pos = (sentiment_encodeing == 1).sum()
  
    print("Positive", pos)
    print("Negative", neg)

    p = [pos, neg, ]
    plt.pie(p,    #giving array
        labels = ['pozitif', 'negatif'], #Correspndingly giving labels
        colors = ['green', 'yellow'],   # Corresponding colors
        explode = (0.15, 0.15),    #How much the gap should me there between the pies
        startangle = 0)  #what start angle should be given
    plt.axis('equal') 
    print(plt.show())
    return sentiment_encodeing