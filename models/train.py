import os
import pandas as pd
import model as model
from model import model_RE
import pickle

path = 'C:/Users/Saman/ml-new/oscer/data/REdata/train.tsv'
train = pd.read_csv(path, sep='\t', names=['sentence' , 'label'])

#Training the model with data
model_RE(train)

# save the model to disk
RE_trained_model = 'finalized_model.sav'
pickle.dump(clf, open(RE_trained_model, 'wb'))