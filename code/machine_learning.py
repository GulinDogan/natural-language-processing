from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from wordcloud import WordCloud 
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

def ml(Train_X, Test_X, Train_Y, Test_Y ) :

    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(Train_X)

    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)

    wd_list =  Tfidf_vect.vocabulary_

    print("Tweetlerdeki kelimeler ve tekrar sayıları:\n", wd_list)

    n_Test_X_Tfidf= normalize(Test_X_Tfidf, norm='max')
    n_Train_X_Tfidf= normalize(Train_X_Tfidf, norm='max')

    Naive = BernoulliNB()
    Naive.fit(n_Train_X_Tfidf,Train_Y)
    predictions_NB = Naive.predict(n_Test_X_Tfidf)
    print("Bernoulli NB  F1 Score -> ", f1_score(Test_Y, predictions_NB, average='macro'))

    cm = confusion_matrix(Test_Y, predictions_NB,  normalize = True)
    print ("Confusion Matrix:\n",cm)

    lr = LogisticRegression(random_state=0).fit(n_Train_X_Tfidf,Train_Y)
    predictions_lr = lr.predict(n_Test_X_Tfidf)
    print("Lojistik Regresyon F1 Score -> ",  f1_score(predictions_lr, Test_Y, average='macro'))

    cm = confusion_matrix(Test_Y, predictions_lr, normalize = True)
    print ("Confusion Matrix:\n",cm)