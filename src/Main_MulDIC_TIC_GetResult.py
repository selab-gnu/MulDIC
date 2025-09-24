# -*- coding: utf-8 -*-

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

import torchvision.transforms as transforms # 이미지 변환 툴
import torchvision
from PIL import Image
from PIL import ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 깨진 이미지 무시하고 로드

seed_val = 0
random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

print(torch.cuda.is_available())
dir_path = os.getcwd()
#file_list = os.listdir(dir_path)
print('Now Directory: ', dir_path)
#print(file_list)
#exit()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class EmbeddingLayer:
    def __init__(self, project, wordSet, embeddingSize, data_type):
        self.wordSet = wordSet
        self.embeddingSize = embeddingSize
        if data_type == 'text':
            try:
                self.embedder = torch.load('src/EmbeddingModel/{}-EmbeddingLayer_text_T-I-C_text.pt'.format(project))
            except FileNotFoundError:
                self.embedder = nn.Embedding(num_embeddings=len(wordSet), embedding_dim=embeddingSize)
                torch.save(self.embedder, 'src/EmbeddingModel/{}-EmbeddingLayer_text_T-I-C_text.pt'.format(project))
        elif data_type == 'code':
            try:                
                self.embedder = torch.load('src/EmbeddingModel/{}-EmbeddingLayer_text_T-I-C_code.pt'.format(project))
            except FileNotFoundError:
                self.embedder = nn.Embedding(num_embeddings=len(wordSet), embedding_dim=embeddingSize)
                torch.save(self.embedder, 'src/EmbeddingModel/{}-EmbeddingLayer_text_T-I-C_code.pt'.format(project))


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

            if len(temp)>=1:
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
        modelPath = 'src/model_MulDIC_TIC/'

        #### Set the project name. ####
        project = 'totalProj'  # totalProj_2000: 16*16 & simple

        modelType = 'multimodal_TIC'
        wordSet_text = None
        wordSet_code = None

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

        print('----{}----{}----'.format(project, modelType))
        train_text, test_text, train_code, test_code, train_img, test_img = self._readFile(project)
        
        print('train_text len: {}'.format(len(train_text)))
        print("=========================")
        print('test len: {}'.format(len(test_text)))
        print("=========================")
        print('train_code len: {}'.format(len(train_code)))
        print("=========================")
        print('test_code len: {}'.format(len(test_code)))
        print("=========================")
        print('train_img len: {}'.format(len(train_img)))
        print("=========================")
        print('test_img len: {}'.format(len(test_img)))
        print("=========================")


        '''
        start1 = time.time()
        trainX_text, trainY_text, trainWordSet_text = self._preprocess(train_text)
        print("train text preprocess time :", time.time() - start1)
        start2 = time.time()
        trainX_code, trainY_code, trainWordSet_code = self._preprocess_code(train_code)
        print("train code preprocess time :", time.time() - start2)

        start3 = time.time()
        testX_text, testY_text, testWordSet_text = self._preprocess(test_text)
        print("test text preprocess time :", time.time() - start3)
        start4 = time.time()
        testX_code, testY_code, testWordSet_code = self._preprocess_code(test_code)
        print("test code preprocess time :", time.time() - start4)
        
        if wordSet_text is None:
            wordSet_text = trainWordSet_text
            wordSet_text.extend(testWordSet_text)
        wordSet_text = sorted(list(wordSet_text))
        print("text wordSet len :", len(wordSet_text))
        
        if wordSet_code is None:
            wordSet_code = trainWordSet_code
            wordSet_code.extend(testWordSet_code)
        wordSet_code = sorted(list(wordSet_code))
        print("code wordSet len :", len(wordSet_code))

        trainLen = np.quantile([len(x[0]) for x in trainX_code], 0.75)   # Third quartile
        testLen = np.quantile([len(x[0]) for x in testX_code], 0.75)
        maxLen = np.max([trainLen, testLen])
        maxLen = int(maxLen)
        #maxLen = 171  # for additional Train
        print("maxLen:", maxLen)
        '''
        
        #embedding_true = True  # 새롭게 임베딩 할 경우 True
        embedding_true = False  # 기존 임베딩 벡터 사용 False
        
        if embedding_true:
            print("=============== embedding_true ===============")
            start5 = time.time()
            trainX_text, trainY_text = self._embedding(project, trainX_text, trainY_text, wordSet_text, labels, maxLen)
            print("train text embedding time :", time.time() - start5)
            start6 = time.time()
            trainX_code, trainY_code = self._embedding_code(project, trainX_code, trainY_code, wordSet_code, labels, maxLen)
            print("train code embedding time :", time.time() - start6)
    
            start7 = time.time()
            testX_text, testY_text = self._embedding(project, testX_text, testY_text, wordSet_text, labels, maxLen)
            print("test text embedding time :", time.time() - start7)
            start8 = time.time()
            testX_code, testY_code = self._embedding_code(project, testX_code, testY_code, wordSet_code, labels, maxLen)
            print("test code embedding time :", time.time() - start8)
            
            print("trainX_text: ", trainX_text.shape)
            print("trainX_code: ", trainX_code.shape)
            print("trainY_text: ", trainY_text.shape)
            print("testX_text: ", testX_text.shape)
            print("testX_code: ", testX_code.shape)
            print("testY_text: ", testY_text.shape)
            
            embedded_data = {
                'trainX_text': trainX_text.numpy(),
                'trainX_code': trainX_code.numpy(),
                'trainY_text': trainY_text.numpy(),
                'testX_text': testX_text.numpy(),
                'testX_code': testX_code.numpy(),
                'testY_text': testY_text.numpy()
            }
            
            #np.savez('src/Embedded_data/embedded_data.npz', **embedded_data)
            np.savez('src/Embedded_data/embedded_data-{}-{}-new.npz'.format(project, modelType), **embedded_data)
            
        else:
            print("=============== embedding_False ===============")
            # 저장된 임베딩된 데이터 로드
            #loaded_data = np.load('embedded_data.npz')
            loaded_data = np.load('src/Embedded_data/embedded_data-{}-{}.npz'.format(project, modelType))
            #loaded_data = np.load('src/Embedded_data/embedded_data-totalProj-multimodal_TIC.npz')
            
            # PyTorch 텐서로 변환
            trainX_text = torch.from_numpy(loaded_data['trainX_text'])
            trainX_code = torch.from_numpy(loaded_data['trainX_code'])
            trainY_text = torch.from_numpy(loaded_data['trainY_text'])
            testX_text = torch.from_numpy(loaded_data['testX_text'])
            testX_code = torch.from_numpy(loaded_data['testX_code'])
            testY_text = torch.from_numpy(loaded_data['testY_text'])
            
            print("trainX_text: ", trainX_text.shape)
            print("trainX_code: ", trainX_code.shape)
            print("trainY_text: ", trainY_text.shape)
            print("testX_text: ", testX_text.shape)
            print("testX_code: ", testX_code.shape)
            print("testY_text: ", testY_text.shape)
      
        
        #self._train(trainX_text, trainY_text, trainX_code, train_img, project, modelPath, modelType, 1000, 256, maxLen, len(labels))
        #self._train(trainX_text, trainY_text, trainX_code, train_img, project, modelPath, modelType, 1000, 512, maxLen, len(labels))
        #self._train(trainX_text, trainY_text, trainX_code, train_img, project, modelPath, modelType, 1000, 1024, maxLen, len(labels))
        # self._train(trainX_text, trainY_text, trainX_code, train_img, project, modelPath, modelType, 50, 256, maxLen, len(labels))
        #self._train(trainX_text, trainY_text, trainX_code, train_img, project, modelPath, modelType, 2000, 256, maxLen, len(labels))

        #self._test(testX_text, testY_text, testX_code, test_img, project, modelPath, modelType)
        self._test_resultExtract(testX_text, testY_text, testX_code, test_img, project, modelPath, modelType)

        wordSet = None

    def _readFile(self, project):
        #### Read the data for that project ####
        df_train_1 = pd.read_csv('multimodal_BothCodeImage_dataset//train_data//train_bothData_bug.csv')
        df_test_1 = pd.read_csv('multimodal_BothCodeImage_dataset//test_data//test_bothData_bug.csv')

        df_train_2 = pd.read_csv('multimodal_BothCodeImage_dataset//train_data//train_bothData_documentation.csv')
        df_test_2 = pd.read_csv('multimodal_BothCodeImage_dataset//test_data//test_bothData_documentation.csv')

        df_train_3 = pd.read_csv('multimodal_BothCodeImage_dataset//train_data//train_bothData_duplicate.csv')
        df_test_3 = pd.read_csv('multimodal_BothCodeImage_dataset//test_data//test_bothData_duplicate.csv')

        df_train_4 = pd.read_csv('multimodal_BothCodeImage_dataset//train_data//train_bothData_enhancement.csv')
        df_test_4 = pd.read_csv('multimodal_BothCodeImage_dataset//test_data//test_bothData_enhancement.csv')

        df_train_5 = pd.read_csv('multimodal_BothCodeImage_dataset//train_data//train_bothData_feature.csv')
        df_test_5 = pd.read_csv('multimodal_BothCodeImage_dataset//test_data//test_bothData_feature.csv')

        df_train_6 = pd.read_csv('multimodal_BothCodeImage_dataset//train_data//train_bothData_good-first-issue.csv')
        df_test_6 = pd.read_csv('multimodal_BothCodeImage_dataset//test_data//test_bothData_good-first-issue.csv')

        df_train_7 = pd.read_csv('multimodal_BothCodeImage_dataset//train_data//train_bothData_help-wanted.csv')
        df_test_7 = pd.read_csv('multimodal_BothCodeImage_dataset//test_data//test_bothData_help-wanted.csv')

        df_train_8 = pd.read_csv('multimodal_BothCodeImage_dataset//train_data//train_bothData_invalid.csv')
        df_test_8 = pd.read_csv('multimodal_BothCodeImage_dataset//test_data//test_bothData_invalid.csv')

        df_train_9 = pd.read_csv('multimodal_BothCodeImage_dataset//train_data//train_bothData_question.csv')
        df_test_9 = pd.read_csv('multimodal_BothCodeImage_dataset//test_data//test_bothData_question.csv')

        df_train_10 = pd.read_csv('multimodal_BothCodeImage_dataset//train_data//train_bothData_wontfix.csv')
        df_test_10 = pd.read_csv('multimodal_BothCodeImage_dataset//test_data//test_bothData_wontfix.csv')


        X_text_train = pd.concat([df_train_1['text'], df_train_2['text'], df_train_3['text'], df_train_4['text'], df_train_5['text'], 
                                  df_train_6['text'], df_train_7['text'], df_train_8['text'], df_train_9['text'], df_train_10['text']], axis=0)
        X_text_test = pd.concat([df_test_1['text'], df_test_2['text'], df_test_3['text'], df_test_4['text'], df_test_5['text'], 
                                  df_test_6['text'], df_test_7['text'], df_test_8['text'], df_test_9['text'], df_test_10['text']], axis=0)
        X_code_train = pd.concat([df_train_1['code'], df_train_2['code'], df_train_3['code'], df_train_4['code'], df_train_5['code'], 
                                  df_train_6['code'], df_train_7['code'], df_train_8['code'], df_train_9['code'], df_train_10['code']], axis=0)
        X_code_test = pd.concat([df_test_1['code'], df_test_2['code'], df_test_3['code'], df_test_4['code'], df_test_5['code'], 
                                  df_test_6['code'], df_test_7['code'], df_test_8['code'], df_test_9['code'], df_test_10['code']], axis=0)

        y_train = pd.concat([df_train_1['final_label'], df_train_2['final_label'], df_train_3['final_label'], df_train_4['final_label'], df_train_5['final_label'], 
                             df_train_6['final_label'], df_train_7['final_label'], df_train_8['final_label'], df_train_9['final_label'], df_train_10['final_label']], axis=0)
        y_test = pd.concat([df_test_1['final_label'], df_test_2['final_label'], df_test_3['final_label'], df_test_4['final_label'], df_test_5['final_label'], 
                            df_test_6['final_label'], df_test_7['final_label'], df_test_8['final_label'], df_test_9['final_label'], df_test_10['final_label']], axis=0)
        # print("================== X Train ==================")
        # print(X_text_train.info())
        # print("================== X Test ==================")
        # print(X_text_test.info())
        # print("================== y Train ==================")
        # print(y_train.info())
        # print("================== y Test ==================")
        # print(y_test.info())
        # exit()
        train_text = pd.concat([X_text_train, y_train], axis=1).values.tolist()
        test_text = pd.concat([X_text_test, y_test], axis=1).values.tolist()
        train_code = pd.concat([X_code_train, y_train], axis=1).values.tolist()
        test_code = pd.concat([X_code_test, y_test], axis=1).values.tolist()

        #### Read the image data for that project ####
        #trans = transforms.Compose([transforms.Resize((258,258)), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        trans = transforms.Compose([transforms.Resize((16,16)), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        #trans = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        #trans = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        trainset = torchvision.datasets.ImageFolder(root='multimodal_BothCodeImage_dataset//train_data',
                                                    transform=trans)
        testset = torchvision.datasets.ImageFolder(root='multimodal_BothCodeImage_dataset//test_data',
                                                    transform=trans)
        train_img = trainset
        test_img = testset
        classes = trainset.classes
        print(classes)

        return train_text, test_text, train_code, test_code, train_img, test_img
    
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
    
    def _preprocess_code(self, data):
        preprocessor = TextPreprocessor('code')
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
        embedder = EmbeddingLayer(project, wordSet, 300, 'text')
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
    
    def _embedding_code(self, project, x, y, wordSet, labels, maxLen):
        embedder = EmbeddingLayer(project, wordSet, 300, 'code')
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

    def _train(self, X, Y, X_code, X_img, project, modelPath, modelType, epoch, batchSize, maxLen, numClass):
        trainer = Trainer(project, modelPath, modelType, epoch, 300, batchSize, maxLen, numClass, 'EmbeddingLayer')
        trainer.fit(X, Y, X_code=X_code, X_img=X_img)

    def _test(self, X, Y, X_code, X_img, project, modelPath, modelType, task=None, k=None):
        Y = Y.detach().cpu().numpy()
        for modelName in os.listdir(modelPath+project):
            classifier = Classifier(project, modelPath, modelName)
            prediction = classifier.classify(X, data_code=X_code, data_img=X_img).detach().cpu().numpy()
            self._evaluate(prediction, Y, project, modelName)
    
    def _test_resultExtract(self, X, Y, X_code, X_img, project, modelPath, modelType, task=None, k=None):
        Y = Y.detach().cpu().numpy()
        #modelName = 'multimodal-EmbeddingLayer-1000.pt'
        modelName = 'multimodal_TIC-EmbeddingLayer-800.pt'  # Check the Epoch Value
        classifier = Classifier(project, modelPath, modelName)
        
        predicts = classifier.classify_resultExtract(X, data_code=X_code, data_img=X_img)
        prediction = np.array(predicts[0].tolist())
        predicted_value = np.array(predicts[1].tolist())
        
        pred_df = pd.DataFrame(prediction, columns=['label'])
        pred_val_df = pd.DataFrame(predicted_value)
        real_val_df = pd.DataFrame(Y, columns=['real_label'])
        
        pred_out = pd.concat([pred_val_df, pred_df], axis=1)
        out_df = pd.concat([pred_out, real_val_df], axis=1)

        #out_df.to_csv('C://Users//MyPC//Desktop//project_github//resultData_extracted//multimodalResult_flutter.csv')
        out_df.to_csv('resultData_extracted/MulDIC-TIC_result.csv')
        print('======================== pred_out ========================')
        print(pred_out)
        print('\n======================== out_df ========================')
        print(out_df)
        exit()

    def _evaluate(self, prediction, real, project, modelName, task=None, k=None):
        evaluator = Evaluator(project, 'src/result/', modelName)
        evaluator.evaluate(prediction, real)

if __name__ == '__main__':
    start = time.time()

    main = MAIN()
    main.run()

    print("time :", time.time() - start)
    