import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pickle
import argparse
from torch.utils.data import Dataset, DataLoader
from student_models import resnet18


TEST_PATH = "./data/test.pkl"
TRAIN_PATH = "./data/train.pkl"
ACC_DATA_PATH = "./results/accuracy-curve-"
NUM_CONFIGS = 5
NUM_CHANNELS = 3
NUM_EPOCHS = 60
PLOT_FREQ = 4
HEIGHT = 224
WIDTH = 224


class TeacherTrainset(Dataset):

    def __init__(self, train_path):
      # data loading
      self.train_path = train_path
      with open(train_path, 'rb') as f:
        teacher_set = pickle.load(f)
        self.x = np.zeros((len(teacher_set), NUM_CHANNELS+1, HEIGHT, WIDTH))
        self.y = np.zeros(len(teacher_set))
        for i,d in enumerate(teacher_set):
            image = d[0][1]
            middle_explanation = np.expand_dims(d[1][1], axis=0)
            label = d[2][1]
            self.x[i][:NUM_CHANNELS] = image
            self.x[i][NUM_CHANNELS] = middle_explanation
            self.y[i] = label
        self.n_samples = len(teacher_set)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples

class TeacherTestset(Dataset):

    def __init__(self, test_path):
      # data loading
      with open(test_path, 'rb') as f:
        teacher_set = pickle.load(f)
        self.x = np.zeros((len(teacher_set), NUM_CHANNELS, HEIGHT, WIDTH))
        self.y = np.zeros(len(teacher_set))
        for i,d in enumerate(teacher_set):
            image = d[0][1]
            label = d[1][1]
            self.x[i] = image
            self.y[i] = label
        self.n_samples = len(teacher_set)

    def __getitem__(self, index):
      return self.x[index], self.y[index]
    
    def __len__(self):
      return self.n_samples

trainset = TeacherTrainset(TRAIN_PATH)
trainloader = DataLoader(dataset=trainset, batch_size=64, shuffle=True, num_workers=2)
testset = TeacherTestset(TEST_PATH)
testloader = DataLoader(dataset=testset, batch_size=64, shuffle=False, num_workers=2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

cross_entropy = nn.CrossEntropyLoss()
def student_loss(output, target, explanation_type='middle-layer', lam=1e-4):
    final = output['final']
    explanation = output[explanation_type]
    predicted_explanation = output['predicted explanation'].squeeze()
    target = target.type(torch.cuda.LongTensor)
    loss = cross_entropy(final, target)
    #print(str(loss) + ', ' + str(torch.dist(explanation, predicted_explanation)))
    loss = loss + lam * torch.dist(explanation, predicted_explanation)
    return loss

def train_model(explanation):
    accuracy_curve = np.zeros((NUM_CONFIGS+1, NUM_EPOCHS // PLOT_FREQ + 1))
    for t in range(5):
        for l in range(-1,NUM_CONFIGS,1):
            net = resnet18()
            optimizer = optim.Adam(net.parameters(), lr=1e-3)
            net.to(device)
              
            # train
            if l == -1:
                lam = 0
            else:
                lam = 10 ** (-2*l) # 1,1e-2,1e-4,...1e-10
            # if l == -1:
            #     lam = 10 ** (-1.5)
            # if l == 0:
            #     lam = 10 ** (-1)
            # if l == 1:
            #     lam = 10 ** (-0.5)
            print('lam: ' + str(lam))

            curr_curve = [0]

            print('trainloader len: ' + str(len(trainloader)))
            plot_index = 1
            for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
                running_loss = 0.0    
                for i, data in enumerate(trainloader):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, teacher_predictions = data[0].to(device), data[1].to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward + backward + optimize
                    student_outputs = net(inputs)
                    loss = student_loss(student_outputs, teacher_predictions, explanation_type=explanation, lam=lam)
                    loss.backward()
                    optimizer.step()
                        
                if (epoch + 1) % PLOT_FREQ == 0:
                    # evaluate accuracy on test set
                    net.eval()
                      
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for data in testloader:
                            images, labels = data[0].to(device), data[1].to(device)
                            outputs = net(images)['final']
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted.cpu().numpy() == labels.cpu().numpy()).sum().item()

                    acc = (100 * correct / total)
                    curr_curve.append(acc)
                    print('Accuracy of the network on the test images: %d %%' % acc)

                    net.train()

            curr_curve = np.array(curr_curve)
            accuracy_curve[l+1] = curr_curve
            print('Finished training model with lambda: ' + str(lam))

            # saving model
            #PATH = F"./models/" + str(explanation) + "/" + str(size) + "/run-" + str(j) + "-student-21-layers-" + str(lam)
            #torch.save(net.state_dict(), PATH)
        np.save(ACC_DATA_PATH + str(t) + '.npy', accuracy_curve)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse explanation type")
    parser.add_argument('explanation_type', type=str, help='A required string argument for the explanation type used to supervise the student training')
    args = parser.parse_args()
    explanation = args.explanation_type
    train_model(explanation)
