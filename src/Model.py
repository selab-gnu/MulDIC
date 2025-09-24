import torch
import torch.nn as nn
from Models import Models

class Model:
    def __init__(self, modelType=None, numClass=None, numCell=None, maxLen=None, test=False, additional_train=False, modelPath=None, project=None):
        if not test:
            self.model = Models.getModel(modelType, numClass, numCell, maxLen).cuda()
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
            self.optimizer22 = torch.optim.Adam(self.model.parameters(), lr=0.0002)
        else:
            self.model = None
                         
        #self.model = nn.DataParallel(self.model, device_ids = [0,1])
        #self.model.cuda()
        
        #additional_train = False  # 기본은 False
        #additional_train = True  # 기본은 False
        if additional_train:
            self.modelPath = modelPath
            self.project = project
            print("------ additional_train 22 ------")
               
            pretrained_state_dict = torch.load('{}{}/multimodal_TIC-EmbeddingLayer-1000.pt'.format(self.modelPath, self.project))
            
            #self.model = nn.DataParallel(self.model, device_ids=[0, 1])
            #self.model.cuda()
            
            #self.model.module.load_state_dict(pretrained_state_dict)
            
            # 모델을 CPU로 이동시키고 DataParallel을 벗겨냅니다.
            self.model = self.model.cpu().module
            self.model.load_state_dict(pretrained_state_dict)
            # 다시 GPU로 이동시킵니다.
            self.model = nn.DataParallel(self.model, device_ids=[0, 1])
            self.model.cuda()
            
            self.model.eval()
            print("------------------------------")
        else:
            print("------ First time 22 ------")
            self.model = nn.DataParallel(self.model, device_ids = [0,1])
            self.model.cuda()
    
    def fit(self, x=None, y=None, last=None, x_img=None, x_code=None):
        if x_img != None and x_code != None:
            pred = self.model(x_t = x, x_c = x_code, x_i = x_img)
            loss = self.criterion(pred, y)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            # self.optimizer22.step()
            if last:
                loss = None
        elif x_img != None:
            if x != None:
                pred = self.model(x_t = x, x_i = x_img)
                loss = self.criterion(pred, y)
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
                if last:
                    loss = None
            elif x == None:
                # print("@@@@@@@@@@@@@ fit 실행!!! @@@@@@@@@@@@@")
                # print(y.shape)
                pred = self.model(x_i = x_img)
                # print("#############")
                # print(pred.shape)
                # print("#############")
                
                loss = self.criterion(pred, y)
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
                if last:
                    loss = None
        elif x_code != None:
            pred = self.model(x_t = x, x_c = x_code)
            loss = self.criterion(pred, y)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            if last:
                loss = None
        else:
            pred = self.model(x)
            loss = self.criterion(pred, y)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()       
            # self.optimizer22.step()
            if last:
                loss = None

    def predict(self, x, x_img=None, x_code=None):
        if x_img != None and x_code != None:
            return self.model(x_t = x, x_c = x_code, x_i = x_img)
        elif x_img != None:
            return self.model(x_t = x, x_i = x_img)
        elif x_code != None:
            return self.model(x_t = x, x_c = x_code)
        else:
            return self.model(x)

    def save(self, path):
        torch.save(self.model, path)

    def load(self, path):
        self.model = torch.load(path)