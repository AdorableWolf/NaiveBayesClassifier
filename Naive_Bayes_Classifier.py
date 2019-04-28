import pandas as pd
import numpy as np
import re
from collections import defaultdict

'''
    Clean the string argument and return the cleaned string.

    Preprocess the string argument - str_arg - such that :
        1. everything apart from letters is excluded
        2. multiple spaces are replaced by single space
        3. str_arg is converted to lower case
'''
def preprocess_String(str_arg):
    cleaned_str = re.sub('[^a-z\s]+',' ',str_arg, flags=re.IGNORECASE)
    cleaned_str=re.sub('(\s+)',' ',cleaned_str) 
    cleaned_str = cleaned_str.lower()

    return cleaned_str

class NaiveBayes: 
        def __init__(self, unique_classes):
            self.classes = unique_classes
        
        def addtoBoW(self, example, dict_index):
            '''
                It simply splits the example on the basis of space as a tokenizer and 
                adds every tokenized word to its corresponding dictionary/BoW
            '''
            if isinstance(example, np.ndarray): example=example[0]
            
            for token_word in example.split():
                self.bow_dicts[dict_index][token_word]+=1
        
        def train(self, dataset, labels):
            ''' 
                This is the training function which will train the Naive Bayes Model 
                i.e compute a BoW for each category/class. 
            '''
            self.examples = dataset
            self.labels = labels
            self.bow_dicts = np.array([defaultdict(lambda:0) for index in range(self.classes.shape[0])])

            if not isinstance(self.examples,np.ndarray): self.examples=np.array(self.examples)
            if not isinstance(self.labels,np.ndarray): self.labels=np.array(self.labels)
            
            #constructing BoW for each category
            for cat_index,cat in enumerate(self.classes):
                all_cat_examples = self.examples[self.labels==cat]

                #get examples preprocessed
                cleaned_examples = [preprocess_String(cat_example) for cat_example in all_cat_examples]
                cleaned_examples = pd.DataFrame(data=cleaned_examples)

                #now costruct BoW of this particular category
                np.apply_along_axis(self.addtoBoW,1,cleaned_examples,cat_index)
            
            '''
            Computing addtional figures to aid the improvement of the model:
                1. prior probability of each class - p(c)
                2. vocabulary |V| 
                3. denominator value of each class - [ count(c) + |V| + 1 ] 
            '''
            prob_classes = np.empty(self.classes.shape[0])
            all_words = []
            cat_word_counts = np.empty(self.classes.shape[0])

            for cat_index,cat in enumerate(self.classes):
                #Calculating prior probability p(c) for each class
                prob_classes[cat_index] = np.sum(self.labels==cat)/float(self.labels.shape[0])

                #Calculating total counts of all the words of each class 
                count = list(self.bow_dicts[cat_index].values())
                cat_word_counts[cat_index] = np.sum(np.array(list(self.bow_dicts[cat_index].values())))+1

                #get all words of this category
                all_words+=self.bow_dicts[cat_index].keys()
            
            #combine all words of every category & make them unique to get vocabulary -V- of entire training set
            self.vocab = np.unique(np.array(all_words))
            self.vocab_length = self.vocab.shape[0]

            #computing denominator value
            denoms=np.array([cat_word_counts[cat_index]+self.vocab_length+1 for cat_index,cat in enumerate(self.classes)])

            self.cats_info=[(self.bow_dicts[cat_index],prob_classes[cat_index],denoms[cat_index]) for cat_index,cat in enumerate(self.classes)]                               
            self.cats_info=np.array(self.cats_info) 

        def getExampleProb(self, test_example):
            likelihood_prob = np.zeros(self.classes.shape[0])

            #finding probability w.r.t each class of the given test example
            for cat_index,cat in enumerate(self.classes):

                for test_token in test_example.split():
                    
                    #get total count of this test token from it's respective training dict to get numerator value
                    test_token_counts = test_token_counts=self.cats_info[cat_index][0].get(test_token,0)+1
                    
                    #now get likelihood of this test_token word  
                    test_token_prob=test_token_counts/float(self.cats_info[cat_index][2])
                    
                    #To prevent underflow!
                    likelihood_prob[cat_index]+=np.log(test_token_prob)
            
            #posterior probility
            post_prob=np.empty(self.classes.shape[0])
            for cat_index,cat in enumerate(self.classes):
                post_prob[cat_index]=likelihood_prob[cat_index]+np.log(self.cats_info[cat_index][1])
            
            return post_prob
        
        def test(self, test_set):
            predictions = [] #to store prediction of each test example
            for example in test_set:
                cleaned_example = preprocess_String(example)

                #get the posterior probability of every example
                post_prob = self.getExampleProb(cleaned_example)

                #pick the max value and map against self.classes
                predictions.append(self.classes[np.argmax(post_prob)])

            return np.array(predictions)