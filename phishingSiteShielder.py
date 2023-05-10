import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix,mean_squared_error,precision_score,recall_score,f1_score
from sklearn.metrics import classification_report, f1_score, recall_score
import os
import pickle

df= pd.read_csv("./Phishing Site Shielder/CSV/phishing_site_urls.csv")
print(df.head())
print(df.shape)

x=df.Label.unique()
print(x)

y=np.array([df[df['Label']=='bad'].count()[0],df[df['Label']=='good'].count()[0]])
print(y)

plt.bar(x,y,color=[ 'red', 'green'])

print(df.info())
print(df.isna().sum())
print(df.URL.duplicated().sum())


df.drop(df[df.URL.duplicated() == True].index, axis = 0, inplace = True)
df.reset_index(drop=True)

df['clean_url']=df.URL.astype(str)
print(df.head())
print(df.URL.duplicated().sum())

# Take only the Alphabets
tok= RegexpTokenizer(r'[A-Za-z0-9]+')
tok.tokenize(df.URL[1])

df.clean_url=df.clean_url.map(lambda x: tok.tokenize(x))
print(df.head())

wnl = WordNetLemmatizer()
df['lem_url'] = df['clean_url'].map(lambda x: [wnl.lemmatize(word) for word in x])
print(df.head())
print(df.isna().sum())

word_vectorizer = TfidfVectorizer(ngram_range=(1,1), max_features =1000)
unigramdataGet= word_vectorizer.fit_transform(df['lem_url'].astype('str'))
unigramdataGet = unigramdataGet.toarray()
vocab = word_vectorizer.get_feature_names_out ()
x=pd.DataFrame(np.round(unigramdataGet, 1), columns=vocab)
x[x>0] = 1

print(x)

# Vectorizing CountVectorizer
cv = CountVectorizer()
feature = cv.fit_transform(df.lem_url.astype('str'))
print(feature)

# It seems that count vec act better
y=df.Label
y=np.where(y=='bad',0,1)
x_train,x_test,y_train,y_test =  train_test_split(feature,y,random_state=42,test_size=0.2,shuffle=True)


def get_accuracy(name, trained_model , x_train, y_train, x_test, y_test):
    tree_predict = trained_model.predict(x_test)
    print("Testing accuracy   :",metrics.accuracy_score(y_test, tree_predict)*100 , "%")
    print("MSE [TEST]          :",mean_squared_error(y_test, tree_predict))


    tree_predict1 = trained_model.predict(x_train)
    print("Training accuracy  :",metrics.accuracy_score(y_train, tree_predict1)*100 ,"%")
    print("MSE [TRAIN]         :",mean_squared_error(y_train, tree_predict1))

    print("precision : ",precision_score(y_test, tree_predict,average='micro'))
    print("recall    : ",recall_score(y_test, tree_predict,average='micro'))
    print("f1_score  : ",f1_score(y_test, tree_predict,average='micro'))


    cf1 = confusion_matrix(y_test,tree_predict)
    sb.heatmap(cf1,annot=True,fmt = '.0f')
    plt.xlabel('prediction')
    plt.ylabel('Actual')
    plt.title(name+ ' Confusion Matrix')
    plt.show()

    print(classification_report(y_train,  trained_model.predict(x_train)))
    print(classification_report(y_test,  trained_model.predict(x_test)))


# SVC performance better got 97% accuracy on test data

# # Naive Bayez
# trained_clf_multinomial_nb = MultinomialNB().fit(x_train, y_train)
# get_accuracy('MultinomialNB',trained_clf_multinomial_nb,x_train, y_train, x_test, y_test)


# # Logistic Reg
# trained_clf_LogisticRegression = LogisticRegression(max_iter=5000).fit(x_train, y_train)
# get_accuracy('LogisticRegression',trained_clf_LogisticRegression,x_train, y_train, x_test, y_test)


# SVC
trained_clf_svc = LinearSVC().fit(x_train, y_train)
get_accuracy('LinearSVC', trained_clf_svc, x_train, y_train, x_test, y_test)


# Create the 'Models' directory if it does not exist
if not os.path.exists('./Phishing Site Shielder/Model'):
    os.makedirs('./Phishing Site Shielder/Model')

# Defining a function to save the model
def save_model(model, filename):
    with open(f'./Phishing Site Shielder/Models/{filename}.obj', 'wb') as f:
        pickle.dump(model, f)

# Defining a function to load the model
def load_model(filename):
    with open(f'./Phishing Site Shielder/Models/{filename}.obj', 'rb') as f:
        return pickle.load(f)

# Saving the models
save_model(trained_clf_svc, 'phishingSiteShielderSvc')

# Loading the models
model = load_model('phishingSiteShielderSvc')

# Evaluating loaded model
get_accuracy('LinearSVC', model, x_train, y_train, x_test, y_test)
