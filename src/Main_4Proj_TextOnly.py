# -*- coding: utf-8 -*-
#Main_4Proj_TextOnly.py

import os
import numpy as np
from collections import Counter
import time
import torch
import torch.nn as nn
import pandas as pd
import random

from TextPreprocessor import TextPreprocessor
from Trainer import Trainer
from Classifier import Classifier
from Logger import Logger

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


seed_val = 0
random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

print(torch.cuda.is_available())
dir_path = os.getcwd()
#file_list = os.listdir(dir_path)
print('now directory: ', dir_path)
#print(file_list)
#exit()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class EmbeddingLayer:
    def __init__(self, project, wordSet, embeddingSize):
        self.wordSet = wordSet
        self.embeddingSize = embeddingSize
        try:
            self.embedder = torch.load('src/EmbeddingModel/{}-EmbeddingLayer_text_TextOnly.pt'.format(project))
        except FileNotFoundError:
            self.embedder = nn.Embedding(num_embeddings=len(wordSet), embedding_dim=embeddingSize)
            torch.save(self.embedder, 'src/EmbeddingModel/{}-EmbeddingLayer_text_TextOnly.pt'.format(project))

    def embedding(self, lines, maxLen):
        emWords = None
        for line in lines:
            temp = []
            for i in range(maxLen):
                try:
                    word = line[i]
                    with torch.no_grad():
                        try:
                            idx = torch.tensor(self.wordSet.index(word))
                            if idx > len(self.wordSet):
                                continue
                            temp.append(idx)
                        except ValueError:
                            continue
                except IndexError:
                    continue
            if len(temp)>1:
                with torch.no_grad():
                    temp = self.embedder(torch.tensor(temp))
                for i in range(maxLen-len(temp)):
                    temp = torch.cat((temp, torch.zeros(1, self.embeddingSize)), axis=0)
                if emWords is None:
                    emWords = temp.view(1, -1, self.embeddingSize)
                else:
                    emWords = torch.cat((emWords, temp.view(1, -1, self.embeddingSize)), axis=0)
        return emWords

class Evaluator:
    def __init__(self, project, resultPath, modelName, task=None, k=None):
        self.resultPath = resultPath
        self.modelName = modelName
        self.project = project

    def evaluate(self, predicted, real, tp='title'):      
        logger = Logger('{}{}/{}.txt'.format(self.resultPath, self.project, self.modelName))
        logger.log('real=> {}\n'.format(str(Counter(real))))
        logger.log('predicted=> {}\n'.format(str(Counter(predicted))))

        precision = precision_score(real, predicted, average='weighted', zero_division=0)
        recall = recall_score(real, predicted, average='weighted', zero_division=0)
        f1 = f1_score(real, predicted, average='weighted', zero_division=0)
        acc = accuracy_score(real, predicted)
        logger.log('-------weighted-------\nprecision: {}\nrecall: {}\nf1-score: {}\naccuracy: {}\n'.format(precision, recall, f1, acc))
        print('================= {} ================='.format(self.modelName))
        print('-------weighted-------\nprecision: {}\nrecall: {}\nf1-score: {}\naccuracy: {}\n'.format(precision, recall, f1, acc))
        print('=====================================')

class MAIN:
    def __init__(self):
        pass

    def run(self):
        #### Set the path to save the model. ####
        # modelPath = './model_base_textOnly/'
        modelPath = 'src/model_base_textOnly/'
        
        #### Set the project name. ####
        #project = 'totalProj'
        project = '4Proj'
        
        modelType = 'cnn'
        wordSet = None

        if project == 'vscode':
            labels = ['bug', 'feature']
        elif project == 'kubernetes':
            labels = ['bug', 'feature']
        elif project == 'flutter':
            labels = ['bug', 'feature']
        elif project == 'roslyn':
            labels = ['bug', 'feature']
        elif project == 'totalProj':
            labels = ['bug', 'documentation', 'duplicate', 'enhancement', 'feature', 
                      'good-first-issue', 'help-wanted', 'invalid', 'question', 'wontfix']
        elif project == '4Proj':
            labels = ['bug', 'documentation', 'duplicate', 'feature', 'help-wanted', 'invalid', 'question']

        print('----{}----{}----'.format(project, modelType))
        train, test = self._readFile(project)
        
        
        print('train len: {}'.format(len(train)))
        print("=========================")
        print('test len: {}'.format(len(test)))
        print("=========================")
        
        ###################################################################
        start1 = time.time()
        trainX, trainY, trainWordSet = self._preprocess(train)
        print("train text preprocess time :", time.time() - start1)
        start2 = time.time()
        testX, testY, testWordSet = self._preprocess(test)
        print("test text preprocess time :", time.time() - start2)
        
        if wordSet is None:
            wordSet = trainWordSet
            wordSet.extend(testWordSet)
        wordSet = sorted(list(wordSet))
        ###################################################################

        maxLen = 100    # 3사분위수 74
        maxLen = int(maxLen)
        print("maxLen:", maxLen)
        # exit()
        
        #embedding_true = True  # 새롭게 임베딩 할 경우 True
        embedding_true = False  # 기존 임베딩 벡터 사용 False
        
        #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        #print(trainX_text)
        #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        #print(trainY_text)
        #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        
        if embedding_true:
            print("=============== embedding_true ===============")
            start3 = time.time()
            trainX, trainY = self._embedding(project, trainX, trainY, wordSet, labels, maxLen)
            print("train text embedding time :", time.time() - start3)
            
            start4 = time.time()
            testX, testY = self._embedding(project, testX, testY, wordSet, labels, maxLen)
            print("test text embedding time :", time.time() - start4)
            
            print("trainX_text: ", trainX.shape)
            print("trainY_text: ", trainY.shape)
            print("testX_text: ", testX.shape)
            print("testY_text: ", testY.shape)
            
            embedded_data = {
                'trainX_text': trainX.numpy(),
                'trainY_text': trainY.numpy(),
                'testX_text': testX.numpy(),
                'testY_text': testY.numpy()
            }
            
            #np.savez('src/Embedded_data/embedded_data.npz', **embedded_data)
            #np.savez('src/Embedded_data/embedded_data-{}-{}-new.npz'.format(project, modelType), **embedded_data)
            np.savez('src/Embedded_data/embedded_data-{}-{}-new.npz'.format(project, modelType), **embedded_data)
            
        else:
            print("=============== embedding_False ===============")
            # 저장된 임베딩된 데이터 로드
            #loaded_data = np.load('embedded_data.npz')
            #loaded_data = np.load('src/Embedded_data/embedded_data-totalProj-multimodal_BinaryCLS_TIC.npz')  # TIC Data
            #loaded_data = np.load('src/Embedded_data/embedded_data-{}-{}.npz'.format(project, modelType))  # TextOnly Data
            #loaded_data = np.load('src/Embedded_data/embedded_data-totalProj-multimodal_BinaryCLS_TIC-downSampling.npz')
            #loaded_data = np.load('src/Embedded_data/embedded_data-totalProj-multimodal_BinaryCLS_TIC-downSampling_2.npz')
            loaded_data = np.load('src/Embedded_data/embedded_data-4Proj-cnn-1.npz')
            
            # PyTorch 텐서로 변환
            trainX_text = torch.from_numpy(loaded_data['trainX_text'])
            #trainX_code = torch.from_numpy(loaded_data['trainX_code'])
            trainY_text = torch.from_numpy(loaded_data['trainY_text'])
            testX_text = torch.from_numpy(loaded_data['testX_text'])
            #testX_code = torch.from_numpy(loaded_data['testX_code'])
            testY_text = torch.from_numpy(loaded_data['testY_text'])
            
            print("trainX_text: ", trainX_text.shape)
            print("trainY_text: ", trainY_text.shape)
            print("testX_text: ", testX_text.shape)
            print("testY_text: ", testY_text.shape)
        #exit()
        
        #self._train(trainX, trainY, project, modelPath, modelType, 1000, 256, maxLen, len(labels))
        #self._test(testX, testY, project, modelPath, modelType)
        self._train(trainX_text, trainY_text, project, modelPath, modelType, 1000, 256, maxLen, len(labels))
        self._test(testX_text, testY_text, project, modelPath, modelType)

        wordSet = None

    def _readFile(self, project):
        #### Read the data for that project ####
        df_train_1 = pd.read_csv('multimodal_BothCodeImage_dataset//train_data_4Proj//train_bothData_bug.csv')
        df_test_1 = pd.read_csv('multimodal_BothCodeImage_dataset//test_data_4Proj//test_bothData_bug.csv')

        df_train_2 = pd.read_csv('multimodal_BothCodeImage_dataset//train_data_4Proj//train_bothData_documentation.csv')
        df_test_2 = pd.read_csv('multimodal_BothCodeImage_dataset//test_data_4Proj//test_bothData_documentation.csv')

        df_train_3 = pd.read_csv('multimodal_BothCodeImage_dataset//train_data_4Proj//train_bothData_duplicate.csv')
        df_test_3 = pd.read_csv('multimodal_BothCodeImage_dataset//test_data_4Proj//test_bothData_duplicate.csv')

        #df_train_4 = pd.read_csv('multimodal_BothCodeImage_dataset//train_data_4Proj//train_bothData_enhancement.csv')
        #df_test_4 = pd.read_csv('multimodal_BothCodeImage_dataset//test_data_4Proj//test_bothData_enhancement.csv')

        df_train_5 = pd.read_csv('multimodal_BothCodeImage_dataset//train_data_4Proj//train_bothData_feature.csv')
        df_test_5 = pd.read_csv('multimodal_BothCodeImage_dataset//test_data_4Proj//test_bothData_feature.csv')

        #df_train_6 = pd.read_csv('multimodal_BothCodeImage_dataset//train_data_4Proj//train_bothData_good-first-issue.csv')
        #df_test_6 = pd.read_csv('multimodal_BothCodeImage_dataset//test_data_4Proj//test_bothData_good-first-issue.csv')

        df_train_7 = pd.read_csv('multimodal_BothCodeImage_dataset//train_data_4Proj//train_bothData_help-wanted.csv')
        df_test_7 = pd.read_csv('multimodal_BothCodeImage_dataset//test_data_4Proj//test_bothData_help-wanted.csv')

        df_train_8 = pd.read_csv('multimodal_BothCodeImage_dataset//train_data_4Proj//train_bothData_invalid.csv')
        df_test_8 = pd.read_csv('multimodal_BothCodeImage_dataset//test_data_4Proj//test_bothData_invalid.csv')

        df_train_9 = pd.read_csv('multimodal_BothCodeImage_dataset//train_data_4Proj//train_bothData_question.csv')
        df_test_9 = pd.read_csv('multimodal_BothCodeImage_dataset//test_data_4Proj//test_bothData_question.csv')

        #df_train_10 = pd.read_csv('multimodal_BothCodeImage_dataset//train_data_4Proj//train_bothData_wontfix.csv')
        #df_test_10 = pd.read_csv('multimodal_BothCodeImage_dataset//test_data_4Proj//test_bothData_wontfix.csv')


        X_text_train = pd.concat([df_train_1['text'], df_train_2['text'], df_train_3['text'], df_train_5['text'], 
                                  df_train_7['text'], df_train_8['text'], df_train_9['text']], axis=0)
        X_text_test = pd.concat([df_test_1['text'], df_test_2['text'], df_test_3['text'], df_test_5['text'], 
                                  df_test_7['text'], df_test_8['text'], df_test_9['text']], axis=0)

        y_train = pd.concat([df_train_1['final_label'], df_train_2['final_label'], df_train_3['final_label'], df_train_5['final_label'], 
                             df_train_7['final_label'], df_train_8['final_label'], df_train_9['final_label']], axis=0)
        y_test = pd.concat([df_test_1['final_label'], df_test_2['final_label'], df_test_3['final_label'], df_test_5['final_label'], 
                            df_test_7['final_label'], df_test_8['final_label'], df_test_9['final_label']], axis=0)

        #print("================== X Train ==================")
        #print(X_text_train.info())
        #print("================== X Test ==================")
        #print(X_text_test.info())
        #print("================== y Train ==================")
        #print(y_train.info())
        #print("================== y Test ==================")
        #print(y_test.info())
        #exit()
        
        train_text = pd.concat([X_text_train, y_train], axis=1).values.tolist()
        test_text = pd.concat([X_text_test, y_test], axis=1).values.tolist()

        return train_text, test_text
    
    def _transform(self, train, test):
        train = [(' '.join(line.split(' ')[1:]).replace('"', ''), line.split(' ')[0], None) for line in train.split('\n')[:-1]]
        test = [(' '.join(line.split(' ')[1:]).replace('"', ''), line.split(' ')[0], None) for line in test.split('\n')[:-1]]
        return train, test

    def _preprocess(self, data):
        preprocessor = TextPreprocessor('tt')
        x = []
        y = []
        for d in data:
            t, l = preprocessor.pp(d)
            if t:
                x.append(t)
                y.append(l)
            else:
                continue
        return x, y, list(preprocessor.wordSet)

    def _embedding(self, project, x, y, wordSet, labels, maxLen):
        embedder = EmbeddingLayer(project, wordSet, 300)
        X = None
        Y = None
        for t, l in zip(x, y):
            emWords = embedder.embedding(t, maxLen)
            l = torch.tensor(labels.index(l)).view(-1)
            if emWords is not None:
                if X is None:
                    X = emWords
                    Y = l
                else:
                    X = torch.cat((X, emWords), dim=0)
                    Y = torch.cat((Y, l), dim=0)
            else:
                continue
        return X, Y

    def _train(self, X, Y, project, modelPath, modelType, epoch, batchSize, maxLen, numClass):
        trainer = Trainer(project, modelPath, modelType, epoch, 300, batchSize, maxLen, numClass, 'EmbeddingLayer')
        trainer.fit(X, Y)

    def _test(self, X, Y, project, modelPath, modelType, task=None, k=None):
        Y = Y.detach().cpu().numpy()
        for modelName in os.listdir(modelPath+project):
            classifier = Classifier(project, modelPath, modelName)
            prediction = classifier.classify(X).detach().cpu().numpy()
            self._evaluate(prediction, Y, project, modelName)

    def _evaluate(self, prediction, real, project, modelName, task=None, k=None):
        # evaluator = Evaluator(project, './result/', modelName)
        evaluator = Evaluator(project, 'src/result/', modelName)
        evaluator.evaluate(prediction, real)

if __name__ == '__main__':
    start = time.time()

    main = MAIN()
    main.run()

    print("time :", time.time() - start)