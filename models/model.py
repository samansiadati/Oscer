#Relation Extraction for Oscer.ai 
#Saman Siadati, Sep 2021

#It's needed to install spacy 
#!pip install -U spacy
    
    
import warnings
import spacy
from spacy import displacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import string
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import pickle

warnings.filterwarnings("ignore")

def model_RE(dataframe):
  
  X = dataframe.sentence
  y = dataframe.label 

  nlp = spacy.load('en_core_web_sm')

  sentence = X[0]
  doc = nlp(sentence)

  print ("{:<15} | {:<8} | {:<15} | {:<20}".format('Token','Relation','Head', 'Children'))
  print ("-" * 70)

  for token in doc:
  # Print the token, dependency nature, head and all dependents of the token
    print ("{:<15} | {:<8} | {:<15} | {:<20}"
         .format(str(token.text), str(token.dep_), str(token.head.text), str([child for child in token.children])))
  
  # Use displayCy to visualize the dependency 
  displacy.render(doc, style='dep', jupyter=True, options={'distance': 120})  


  sent = nlp.create_pipe('sentencizer')
  nlp.add_pipe(sent, before='parser')

  stopwords = list(STOP_WORDS)
  print(stopwords)

  #Tokenization

  punct = string.punctuation

  # cleaning function
  def text_data_cleaning(sentence):
    doc = nlp(sentence)
    
    tokens = []
    for token in doc:
        if token.lemma_ != "-PRON-":
            temp = token.lemma_.lower().strip()
        else:
            temp = token.lower_
        tokens.append(temp)
    
    cleaned_tokens = []
    for token in tokens:
        if token not in stopwords and token not in punct:
            cleaned_tokens.append(token)
    return cleaned_tokens
  
  #Vectorization Feature Engineering (TF-IDF)
  tfidf = TfidfVectorizer(tokenizer = text_data_cleaning, lowercase=True)


  #split data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

  #classification pipeline
  classifier = LinearSVC()
  clf = Pipeline([('tfidf', tfidf), ('clf', classifier)])

  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)

  print(classification_report(y_test, y_pred))

  confusion_matrix(y_test, y_pred)

  return


