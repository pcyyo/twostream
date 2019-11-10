from __future__ import print_function, division
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import sys
sys.path.append(r'/home/chunyan/two_stream/data')
import loader
import time
import copy
import torchvision.models as models
import torch.utils.model_zoo
from torchvision import transforms

'''
Function for training of model
This function can be modified accordingly to include validation set
'''


def train_model(spat_model, spat_criterion, spat_optimizer, spat_scheduler, temp_model,
                temp_criterion, temp_optimizer, temp_scheduler, num_epochs):
    since = time.time()

    best_spat_wts = spat_model.state_dict()
    best_temp_wts = temp_model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train','val']:  # can also uinclude 'val' mode for validation
            if phase == 'train':
                spat_scheduler.step()  # to reduce the learning rate at a scheduled rate
                temp_scheduler.step()
                spat_model.train()
                temp_model.train()
            else:
                spat_model.train(False)
                temp_model.train(False)

            spat_epoch_loss = 0.0
            temp_epoch_loss = 0.0
            corrects=0
            s_corrects = 0
            t_corrects = 0
            # Iterate over data.
            for data in fullloader[phase]:
                # get the inputs
                #                 print("loading the data")
                spat_data, temp_data, labels = data
                # wrap them in Variable
                if use_gpu:
                    spat_data = Variable(spat_data.cuda())
                    temp_data = Variable(temp_data.cuda())
                    labels = Variable(labels.cuda())
                else:
                    spat_data = Variable(spat_data)
                    temp_data = Variable(temp_data)
                    labels = Variable(labels)

                # zero the parameter gradients
                spat_optimizer.zero_grad()
                temp_optimizer.zero_grad()

                # forward
                try:
                    spat_out = spat_model(spat_data)
                    temp_out = temp_model(temp_data)
                except RuntimeError as exception:
                    if "out of memory" in str(exception):
                        print("WARNING: out of memory")
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        raise exception

                outputs = spat_out + temp_out
                _, s_pred = torch.max(spat_out, 1)
                _, t_pred = torch.max(temp_out, 1)
                _, preds = torch.max(outputs, 1)
                # fusion of softmax_scores
                spat_loss = spat_criterion(spat_out, labels)
                temp_loss = temp_criterion(temp_out, labels)

                if phase == 'train':
                    # backward and optimize for training mode
                    spat_loss.backward()
                    spat_optimizer.step()
                    temp_loss.backward()
                    temp_optimizer.step()

                # finding epoch losses and accuracies for fusion
                spat_epoch_loss += spat_loss.item() * spat_data.size(0)
                temp_epoch_loss += temp_loss.item() * temp_data.size(0)

                s_corrects += torch.sum(s_pred == labels.data)
                t_corrects += torch.sum(t_pred == labels.data)
                corrects += torch.sum(preds == labels.data)

            phase_size = len(fullloader[phase]) * fullloader[phase].batch_size
            spat_epoch_loss = spat_epoch_loss / phase_size
            temp_epoch_loss = temp_epoch_loss / phase_size
            accuracy = float(corrects) / phase_size
            s_accuracy = float(s_corrects) / phase_size
            t_accuracy = float(t_corrects) / phase_size

            print('Spatial Loss: {:.4f}   Temporal Loss: {:.4f}   s_Accuracy: {:.2f} t_Accuracy: {:.2f} Accuracy : {:.2f}'.format(
                spat_epoch_loss, temp_epoch_loss, s_accuracy, t_accuracy, accuracy))

            if phase == 'val' and accuracy > best_acc:
                best_acc = accuracy
                best_spat_wts = spat_model.state_dict()
                best_temp_wts = temp_model.state_dict()

            del (spat_data)
            del (temp_data)
            del (labels)
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    spat_model.load_state_dict(best_spat_wts)
    temp_model.load_state_dict(best_temp_wts)
    torch.save(spat_model.state_dict(), 'pre_spat_params.pkl')  # save only the parameters
    torch.save(temp_model.state_dict(), 'pre_temp_params.pkl')  # save only the parameters
    return [spat_model, temp_model]


# loading the train and test data
'''
change here for loading data for spatial loader, temporal loader
'''
data_loader = loader.spatio_temporal_dataloader(BATCH_SIZE=64, num_workers=8, in_channel=15,
                                                    spatial_path='../../share2/ucf-data/jpegs_256/',
                                                    # path for spatial data
                                                    temp_path='../../share2/ucf-data/tvl1_flow/',  # path for temporal data
                                                    ucf_list='../../share2/ucf-data/UCF-101-list/',
                                                    ucf_split='01',
                                                    train_transform=transforms.Compose([
                                                        transforms.RandomCrop(224),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                             std=[0.229, 0.224, 0.225])
                                                    ]),
                                                    val_transform=transforms.Compose([
                                                        transforms.Resize([224, 224]),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                             std=[0.229, 0.224, 0.225])
                                                    ]))
train_loader, test_loader, test_video = data_loader.run()
'''
appending train-loader and test loader for training the model
'''
fullloader = {}
fullloader['train'] = train_loader
fullloader['val'] = test_loader


'''
model
'''
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(30, 96, kernel_size=7, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(5*5*512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 101),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 5*5*512)
        out = self.classifier(x)
        return out

spat_model = CNN();
#spat_model.load_state_dict(torch.load('spat_params.pkl'))
temp_model = CNN();
#temp_model.load_state_dict(torch.load('temp_params.pkl'))
'''
if system has cuda 
'''
use_gpu = torch.cuda.is_available()

if use_gpu:
    spat_model = spat_model.cuda()
    temp_model = temp_model.cuda()

spat_criterion = nn.CrossEntropyLoss()
temp_criterion = nn.CrossEntropyLoss()

spat_optimizer = optim.SGD(spat_model.parameters(), lr=0.01, momentum=0.9)
temp_optimizer = optim.SGD(temp_model.parameters(), lr=0.01, momentum=0.9)

# Decay LR by a factor of 0.1 every 10 epochs
spat_scheduler = lr_scheduler.StepLR(spat_optimizer, step_size=10, gamma=0.1)
temp_scheduler = lr_scheduler.StepLR(temp_optimizer, step_size=10, gamma=0.1)

[spat_model, temp_model] = train_model(spat_model, spat_criterion, spat_optimizer, spat_scheduler,
                                       temp_model, temp_criterion, temp_optimizer, temp_scheduler, num_epochs = 30)