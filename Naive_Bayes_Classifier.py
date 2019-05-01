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
        
        def addtoDict(self, abstract, dict_index):
            '''
                It simply splits the abstract on the basis of space as a tokenizer and 
                adds every tokenized word to its corresponding dictionary
            '''
            if isinstance(abstract, np.ndarray): abstract=abstract[0]
            
            for token_word in abstract.split():
                self.class_dicts[dict_index][token_word]+=1
        
        def train(self, dataset, labels):
            ''' 
                This is the training function which will train the Naive Bayes Model 
                i.e compute a Dictionary for each category/class. 
            '''
            self.abstracts = dataset
            self.labels = labels
            self.class_dicts = np.array([defaultdict(lambda:0) for index in range(self.classes.shape[0])])

            if not isinstance(self.abstracts,np.ndarray): self.abstracts=np.array(self.abstracts)
            if not isinstance(self.labels,np.ndarray): self.labels=np.array(self.labels)
            
            #constructing Dictionary for each category
            for class_index,class_ in enumerate(self.classes):
                all_class_examples = self.abstracts[self.labels==class_]

                #get the abstracts preprocessed
                cleaned_abstracts = [preprocess_String(class_example) for class_example in all_class_examples]
                cleaned_abstracts = pd.DataFrame(data=cleaned_abstracts)

                #now costruct Dictionary of this particular category
                np.apply_along_axis(self.addtoDict,1,cleaned_abstracts,class_index)
            
            '''
            Computing addtional figures to aid the improvement of the model:
                1. prior probability of each class - p(c)
                2. vocabulary |V| 
                3. denominator value of each class - [ count(c) + |V| + 1 ] 
            '''
            prob_classes = np.empty(self.classes.shape[0])
            all_words = []
            class_word_counts = np.empty(self.classes.shape[0])

            for class_index,class_ in enumerate(self.classes):
                #Calculating prior probability p(c) for each class
                prob_classes[class_index] = np.sum(self.labels==class_)/float(self.labels.shape[0])

                #Calculating total counts of all the words of each class 
                count = list(self.class_dicts[class_index].values())
                class_word_counts[class_index] = np.sum(np.array(list(self.class_dicts[class_index].values())))+1

                #get all words of this category
                all_words+=self.class_dicts[class_index].keys()
            
            #combine all words of every category & make them unique to get vocabulary -V- of entire training set
            self.vocab = np.unique(np.array(all_words))
            self.vocab_length = self.vocab.shape[0]

            #computing denominator value
            denoms=np.array([class_word_counts[class_index]+self.vocab_length+1 for class_index,class_ in enumerate(self.classes)])

            self.class_info=[(self.class_dicts[class_index],prob_classes[class_index],denoms[class_index]) for class_index,class_ in enumerate(self.classes)]                               
            self.class_info=np.array(self.class_info) 

        def getExampleProb(self, test_example):
            likelihood_prob = np.zeros(self.classes.shape[0])

            #finding probability w.r.t each class of the given test abstract
            for class_index,class_ in enumerate(self.classes):

                for test_token in test_example.split():
                    
                    #get total count of this test token from it's respective training dict to get numerator value
                    test_token_counts = test_token_counts=self.class_info[class_index][0].get(test_token,0)+1
                    
                    #now get likelihood of this test_token word  
                    test_token_prob=test_token_counts/float(self.class_info[class_index][2])
                    
                    #To prevent underflow!
                    likelihood_prob[class_index]+=np.log(test_token_prob)
            
            #posterior probility
            post_prob=np.empty(self.classes.shape[0])
            for class_index,class_ in enumerate(self.classes):
                post_prob[class_index]=likelihood_prob[class_index]+np.log(self.class_info[class_index][1])
            
            return post_prob
        
        def test(self, test_set):
            predictions = [] #to store prediction of each test abstract
            for abstract in test_set:
                cleaned_abstract = preprocess_String(abstract)

                #get the posterior probability of every example
                post_prob = self.getExampleProb(cleaned_abstract)

                #pick the max value and map against self.classes
                predictions.append(self.classes[np.argmax(post_prob)])

            return np.array(predictions)