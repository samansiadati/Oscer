import pandas as pd
import pickle
from sklearn.svm import LinearSVC


path = 'C:/Users/Saman/ml-new/oscer/data/REdata/train.tsv'
train = pd.read_csv(path, sep='\t', names=['sentence' , 'label'])


# load the model from disk
loaded_model = pickle.load(open(RE_trained_model, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)
