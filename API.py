import os

import re
import pandas as pd
import numpy as np
import random
import pickle 
from configparser import ConfigParser

from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.naive_bayes import MultinomialNB

import nltk
import email

nltk.download('stopwords')
from nltk.corpus import stopwords

configuration = ConfigParser()
configuration.read("config.ini")

class API:
    def __init__(self, emailFile = "emails.csv", fraudEmail = "fradulent_emails.txt"):
        # Params for bert model and tokenization
        self.Nsamp = 2000 # number of samples to generate in each class - 'spam', 'not spam'
        self.maxtokens = 200 # the maximum number of tokens per document
        self.maxtokenlen = 100 # the maximum length of each token
        self.stopwords = stopwords.words('english')
        self.emailPath = emailFile
        self.filePath = fraudEmail
        self.modelNB = None

    def initialize_API(self):
        print("Checking if Trained Model Available...")
        if os.path.exists(os.path.join(configuration['DIRECTORIES']['model'],"modelNB.pickle")):
            with open('modelNB.pickle', 'rb') as handle:
                print("Model Found, Loading Model...")
                self.modelNB = pickle.load(handle)
            return
        elif os.path.exists(os.path.join(configuration['DIRECTORIES']['files'],"train-features.pickle")) and os.path.exists(os.path.join(configuration['DIRECTORIES']['files'],"train-labels.pickle")) and os.path.exists(os.path.join(configuration['DIRECTORIES']['files'],"test-features.pickle")) and os.path.exists(os.path.join(configuration['DIRECTORIES']['files'],"test-labels.pickle")):
            print("Processed Data Found, Training Model...")
            
            self.__train()
        else:
            print("Initializing Process...")
            print("Processing Data...")            
            self.__processEmailData()
            print("Training Model...")
            self.__train()
    def __train(self):
        with open(os.path.join(configuration['DIRECTORIES']['files'],'train-features.pickle'), 'rb') as handle:
            train_x = pickle.load(handle)
        with open(os.path.join(configuration['DIRECTORIES']['files'],'train-labels.pickle'), 'rb') as handle:
            train_y = pickle.load(handle)
        with open(os.path.join(configuration['DIRECTORIES']['files'],'test-features.pickle'), 'rb') as handle:
            test_x = pickle.load(handle)
        with open(os.path.join(configuration['DIRECTORIES']['files'],'test-labels.pickle'), 'rb') as handle:
            test_y = pickle.load(handle)

        train_features = self.__features_transform(train_x)
        test_features = self.__features_transform(test_x)

        self.modelNB=MultinomialNB()
        self.modelNB.fit(train_features,train_y)
        predicted_class_NB=self.modelNB.predict(test_features)
        #self.__model_assessment(test_y,predicted_class_NB)
        with open(os.path.join(configuration['DIRECTORIES']['model'],'modelNB.pickle'), 'wb') as handle:
            pickle.dump(self.modelNB, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    def predict(self, msg):
        if self.modelNB is None:
            with open(os.path.join(configuration['DIRECTORIES']['model'],'modelNB.pickle'), 'rb') as handle:
                self.modelNB = pickle.load(handle)
        
        test = self.__features_transform(msg)
        return self.modelNB.predict(test)


    def __processEmailData(self):
        print("Reading legitimate emails...")
        # Read Ham emails 
        emails = pd.read_csv(os.path.join(configuration['DIRECTORIES']['raw'],self.emailPath))
        bodies = self.__extract_messages(emails)
        self.bodies_df = pd.DataFrame(random.sample(bodies, 10000))
        # Read SPAM emails
        print("Reading fraudulent emails...")
        with open(os.path.join(configuration['DIRECTORIES']['raw'],self.filePath), 'r',encoding="latin1") as file:
            data = file.read()

        fraud_emails = data.split("From r")
        
        fraud_bodies = self.__extract_messages(pd.DataFrame(fraud_emails,columns=["message"],dtype=str))
        self.fraud_bodies_df = pd.DataFrame(fraud_bodies[1:])
        # Convert everything to lower-case, truncate to maxtokens and truncate each token to maxtokenlen
        print("Processing legitimate emails...")
        EnronEmails = self.bodies_df.iloc[:,0].apply(self.__tokenize)
        EnronEmails = EnronEmails.apply(self.__stop_word_removal)
        EnronEmails = EnronEmails.apply(self.__reg_expressions)
        EnronEmails = EnronEmails.sample(self.Nsamp)
        #EnronEmails = EnronEmails.apply(self.__remove_empty)
        print("Processing fraudulent emails...")
        SpamEmails = self.fraud_bodies_df.iloc[:,0].apply(self.__tokenize)
        SpamEmails = SpamEmails.apply(self.__stop_word_removal)
        SpamEmails = SpamEmails.apply(self.__reg_expressions)
        SpamEmails = SpamEmails.sample(self.Nsamp)
        #SpamEmails = SpamEmails.apply(self.__remove_empty)

        
        print("Labeling data...")
        raw_data = pd.concat([SpamEmails,EnronEmails], axis=0).values
        # corresponding labels
        Categories = ['spam','notspam']
        header = ([1]*self.Nsamp)
        header.extend(([0]*self.Nsamp))

        raw_data, header = self.__unison_shuffle(raw_data, header)
        # split into independent 70% training and 30% testing sets
        idx = int(0.7*raw_data.shape[0])

        print("Train Test Splitting...")
        # 70% of data for training
        self.train_x, self.train_y = self.__convert_data(raw_data[:idx],header[:idx])
        # remaining 30% for testing
        self.test_x, self.test_y = self.__convert_data(raw_data[idx:],header[idx:])  
        
        
        sz = np.where(self.train_x == [''])
        self.train_x = np.delete(self.train_x, sz[0])
        self.train_y = np.delete(self.train_y, sz[0])

        sz = np.where(self.test_x == [''])
        self.test_x = np.delete(self.test_x, sz[0])
        self.test_y = np.delete(self.test_y, sz[0])

        #print(self.train_x)
        print("Storing Data...")
        with open(os.path.join(configuration['DIRECTORIES']['files'],'train-features.pickle'), 'wb') as handle:
            pickle.dump(self.train_x, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(configuration['DIRECTORIES']['files'],'train-labels.pickle'), 'wb') as handle:
            pickle.dump(self.train_y, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(configuration['DIRECTORIES']['files'],'test-features.pickle'), 'wb') as handle:
            pickle.dump(self.test_x, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(configuration['DIRECTORIES']['files'],'test-labels.pickle'), 'wb') as handle:
            pickle.dump(self.test_y, handle, protocol=pickle.HIGHEST_PROTOCOL)



        #np.save("train-features.npy", self.train_x)
        #np.save("train-lables.npy", self.train_y)
        #np.save("test-features.npy", self.test_x)
        #np.save("test-labels.npy", self.test_y)
        

    
    #function which takes in y test value and y predicted value and prints the associated model performance metrics
    def __model_assessment(self, y_test,predicted_class):
        print('confusion matrix')
        print(confusion_matrix(y_test,predicted_class))
        print('accuracy')
        print(accuracy_score(y_test,predicted_class))
        print('precision')
        print(precision_score(y_test,predicted_class,pos_label=1))
        print('recall')
        print(recall_score(y_test,predicted_class,pos_label=1))
        print('f-Score')
        print(f1_score(y_test,predicted_class,pos_label=1))
        print('AUC')
        print(roc_auc_score(np.where(y_test==1,1,0),np.where(predicted_class==1,1,0)))
        #plt.matshow(confusion_matrix(y_test, predicted_class), cmap=plt.cm.binary, interpolation='nearest')
        #plt.title('confusion matrix')
        #plt.colorbar()
        #plt.ylabel('expected label')
        #plt.xlabel('predicted label')

    

    def __features_transform(self, mail):
        #get the bag of words for the mail text
        with open(os.path.join(configuration['DIRECTORIES']['files'],'train-features.pickle'), 'rb') as handle:
            train_x = pickle.load(handle)
        #print(train_x.shape)
        bow_transformer = CountVectorizer(analyzer=self.__split_into_lemmas).fit(train_x)
        #print(len(bow_transformer.vocabulary_))
        messages_bow = bow_transformer.transform(mail)
        #print sparsity value
        #print('sparse matrix shape:', messages_bow.shape)
        #print('number of non-zeros:', messages_bow.nnz) 
        #print('sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1])))
        #apply the TF-IDF transform to the output of BOW
        tfidf_transformer = TfidfTransformer().fit(messages_bow)
        messages_tfidf = tfidf_transformer.transform(messages_bow)
        #print(messages_tfidf.shape)
        #return result of transforms
        return messages_tfidf


    def __split_into_lemmas(self, message):
        #print(type(message))
        #if type(message) is 'str':
        #    message = message.lower()
        #else:
        message = message[0].lower()
        words = TextBlob(message).words
        # for each word, take its "base form" = lemma 
        return [word.lemma for word in words]

    def __extract_messages(self, df):
        messages = []
        for item in df["message"]:
            # Return a message object structure from a string
            e = email.message_from_string(item)    
            # get message body  
            message_body = e.get_payload()
            messages.append(message_body)
        print("Successfully retrieved message body from e-mails!")
        return messages

    def __unison_shuffle(self, a, b):
        p = np.random.permutation(len(b))
        data = a[p]
        header = np.asarray(b)[p]
        return data, header


    def __convert_data(self, raw_data,header):
        converted_data, labels = [], []
        for i in range(raw_data.shape[0]):
            out = ' '.join(raw_data[i])
            converted_data.append(out)
            labels.append(header[i])
            #print(i)
        converted_data = np.array(converted_data, dtype=object)[:, np.newaxis]
        
        return converted_data, np.array(labels)



    def __tokenize(self, row):
        if row is None or row is '':
            tokens = ""
        else:
            try:
                tokens = row.split(" ")[:self.maxtokens]
            except:
                tokens=""
        return tokens
    def __reg_expressions(self, row):
        tokens = []
        try:
            for token in row:
                token = token.lower()
                token = re.sub(r'[\W\d]', " ", token)
                token = token[:self.maxtokenlen] # truncate token
                if token != '':
                    tokens.append(token)
        except:
            token = ""
            tokens.append(token)
        return tokens

    def __stop_word_removal(self, row):
        token = [token for token in row if token not in self.stopwords]
        token = list(filter(None, token))
        return token