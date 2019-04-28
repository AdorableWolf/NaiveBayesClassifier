import pandas as pd
import numpy as np
import re
from collections import defaultdict
from Naive_Bayes_Classifier import NaiveBayes

#Read data from the .csv flies respectively.
training_set = pd.read_csv('trg.csv')
test_set = pd.read_csv('tst.csv')

#Extract the individual columns in the data into labels(classes) and data(abstract)
labels = training_set['class'].values
taining_data = training_set['abstract'].values

#Extract the abstract column with from the testing set.
testing_data = test_set['abstract'].values

#intanstiate a NaiveBayes Classifier object
classes = np.unique(labels)

#train the model
nb = NaiveBayes(classes)
nb.train(taining_data,labels)

#Classify the testing set.
prob_classes = nb.test(testing_data)

#Convert the predictions to .csv file and save.
kaggle_df = pd.DataFrame(data=np.column_stack([test_set["id"].values,prob_classes]),columns=["id","class"])
kaggle_df.to_csv("./predictions.csv",index=False)

print ('Predcitions Generated and saved to predictions.csv')