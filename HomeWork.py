import PIL
from torchvision import transforms
import time
import torch
import torch.nn as nn
import math
import numpy as np
import torch

steps =5

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(3,5,kernel_size=3),
                                             nn.Conv2d(5, 5, kernel_size=2,stride=2),
                                             nn.MaxPool2d(kernel_size=2),
                                             nn.Dropout(0.3),
                                             nn.Conv2d(5,12,kernel_size=2),
                                             nn.Conv2d(12, 12, kernel_size=2, stride=2),
                                             nn.MaxPool2d(kernel_size=2),
                                             nn.Dropout(0.2),
                                             nn.Conv2d(12,24,kernel_size=2),
                                             nn.Conv2d(24, 24, kernel_size=2, stride=2),
                                             nn.MaxPool2d(kernel_size=2),
                                             nn.Dropout(0.2),
                                             nn.Conv2d(24, 30, kernel_size=2),
                                             nn.Flatten(),
                                             nn.GELU(),
                                        nn.Linear(120,32),
                                        nn.GELU(),
                                        nn.Linear(32, 2),
                                             nn.Softmax(dim=1))

    def forward(self, input):
        return self.cnn(input)

def Train( device='cuda'):
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.RandomHorizontalFlip(0.5),
                                transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
    criterion = nn.CrossEntropyLoss()
    model = CNN().to(device)
    optimizer = torch.optim.RAdam(model.parameters(),lr=5e-4)
    for _ in range(steps):
        loss_average = 0
        count =0
        FaceAdder, HelmetAdder = 1, 1
        correctness=0
        for i in range(638):
            try:
                count+=1
                if i % 2 == 0:
                    url = './helmet/Helmet_' + str(HelmetAdder) + '.jpg'
                    HelmetAdder += 1
                    Answer = torch.tensor([0]).to(device)
                else:
                    url = './Face/Face_' + str(FaceAdder) + '.jpg'
                    FaceAdder += 1
                    Answer = torch.tensor([1]).to(device)
                Img = PIL.Image.open(url)
                Img = Img.resize((256, 256))
                Img = Img.convert("RGB")
                Img = trans(Img).float().to(device)
                predict = model(Img.reshape(-1, 3, 256, 256))
                loss = criterion(predict, Answer)
                optimizer.zero_grad()
                loss.backward()
                loss_average += loss.item()
                optimizer.step()
                if torch.argmax(predict, dim=1).item() == Answer:
                    correctness += 1
            except:
                pass
        torch.save(model.state_dict(),'./saved.pt')
        print(loss_average / count,correctness/count*100)


def Value( device='cuda'):
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.RandomHorizontalFlip(0.5),
                                transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
    model = CNN().to(device)
    saved = torch.load('./saved.pt')
    model.load_state_dict(saved)
    model.eval()
    count=0
    FaceAdder, HelmetAdder = 1, 1
    correctness = 0
    for i in range(638):
        try:
            if i % 2 == 0:
                url = './helmet/Helmet_' + str(HelmetAdder) + '.jpg'
                HelmetAdder += 1
                Answer = torch.zeros(1, 2).to(device)
                Answer[0][0] += 1
            else:
                url = './Face/Face_' + str(FaceAdder) + '.jpg'
                FaceAdder += 1
                Answer = torch.zeros(1, 2).to(device)
                Answer[0][1] += 1

            Img = PIL.Image.open(url)

            Img = Img.resize((256, 256))
            Img = Img.convert("RGB")
            Img = trans(Img).float().to(device)
            predict = model(Img.reshape(-1, 3, 256, 256))
            count +=1
            if torch.argmax(predict,dim=1).item() == torch.argmax(Answer,dim=1).item():
                correctness+=1
        except:
            pass
    print(correctness/count*100,"%")

if __name__ == '__main__':
    Train()
    Value()
