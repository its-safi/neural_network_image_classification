# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 21:02:59 2021

@author: safiu
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import shutil 

base_path = os.path.join(os.getcwd(), 'input')

try:
    os.mkdir(os.path.join(base_path , 'testset'))
except OSError as error:
    print('ALREADY EXISTING FOLDER' )

# re-arrange the dataset
fl = open(os.path.join(base_path , ('meta/meta/classes.txt')))
cls = fl.readline()
while(cls):
    cls=cls.strip()
    os.mkdir(os.path.join(os.path.join(base_path , 'testset') , cls))
    cls = fl.readline()
    
# Moving test files to testset/, train files will be left.
testfile = open(os.path.join(base_path , ('meta/meta/test.txt')))
img = testfile.readline().strip()
while(img):
    cls = img.split('/')[0]
    src = os.path.join(os.path.join(base_path , 'images') , img.split('/')[0] , img.split('/')[1]) + '.jpg'
    dst = os.path.join(os.path.join(base_path , 'testset') , cls)
    shutil.move(src, dst)
    print(f'\r{img}',end='')
    img = testfile.readline().strip()
    


##############PART 2 MODEL 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from efficientnet_pytorch import EfficientNet
import os
base_path = os.path.join(os.getcwd(), 'input')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1),
        torchvision.transforms.RandomAffine(15),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(15),
        torchvision.transforms.Resize((224,224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


valid_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

train_img_path=os.path.join(base_path , 'images')
test_img_path=os.path.join(base_path , 'testset')
train_dataset = torchvision.datasets.ImageFolder(train_img_path,transform=train_transforms)
valid_dataset = torchvision.datasets.ImageFolder(test_img_path,transform=valid_transforms)

batch_size = 8

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size,shuffle=True,num_workers=4,pin_memory=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size,shuffle=False,num_workers=4,pin_memory=True)

nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

def visualize_images(dataloader):
    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    figure, ax = plt.subplots(nrows=3, ncols=3, figsize=(12, 14))
    classes = list(dataloader.dataset.class_to_idx.keys())
    img_no = 0
    for images,labels in dataloader:
        for i in range(3):
            for j in range(3):
                img = np.array(images[img_no]).transpose(1,2,0)
                lbl = labels[img_no]

                ax[i,j].imshow((img*std) + mean)
                ax[i,j].set_title(classes[lbl])
                ax[i,j].set_axis_off()
                img_no+=1
        break


##Uncomment to visualize images
#visualize_images(train_loader)
#visualize_images(valid_loader)

model = EfficientNet.from_pretrained('efficientnet-b5')



# Freeze first few layers. try different values
for i,param in enumerate(model.parameters()):
    if i<=300:
        param.requires_grad=False

model._dropout = torch.nn.Dropout(0.5)
model._fc = torch.nn.Linear(2048,101)

#Takes nearly 35 minutes to run. Need not run each time. 
from torch_lr_finder import LRFinder
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
lr_finder = LRFinder(model, optimizer, criterion, device="cpu")
lr_finder.range_test(train_loader, end_lr=0.001, num_iter=25)
lr_finder.plot()
lr_finder.reset()

#LR suggestion: steepest gradient
Suggested LR: 1.47E-04
0.000146779926762207

#To run tensorboard
#tensorboard --logdir=summaries

cuda = True
epochs = 25
model_name = 'effnetB5.pt'
optimizer = torch.optim.Adam(model.parameters(),lr=1.47e-4,weight_decay=0.001)
criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.1,patience=2,verbose=True)

writer = SummaryWriter() # For Tensorboard
early_stop_count=0
ES_patience=5
best = 0.0
if cuda:
    model.cuda()
    
for epoch in range(epochs):
    
    # Training
    model.train()
    correct = 0
    train_loss = 0.0
    tbar = tqdm(train_loader, desc = 'Training', position=0, leave=True)
    for i,(inp,lbl) in enumerate(tbar):
        optimizer.zero_grad()
        if cuda:
            inp,lbl = inp.cuda(),lbl.cuda()
        out = model(inp)
        loss = criterion(out,lbl)
        train_loss += loss
        out = out.argmax(dim=1)
        correct += (out == lbl).sum().item()
        loss.backward()
        optimizer.step()
        tbar.set_description(f"Epoch: {epoch+1}, loss: {loss.item():.5f}, acc: {100.0*correct/((i+1)*train_loader.batch_size):.4f}%")
    train_acc = 100.0*correct/len(train_loader.dataset)
    train_loss /= (len(train_loader.dataset)/batch_size)

    #Validation
    model.eval()
    with torch.no_grad():
        correct = 0
        val_loss = 0.0
        vbar = tqdm(valid_loader, desc = 'Validation', position=0, leave=True)
        for i,(inp,lbl) in enumerate(vbar):
            if cuda:
                inp,lbl = inp.cuda(),lbl.cuda()
            out = model(inp)
            val_loss += criterion(out,lbl)
            out = out.argmax(dim=1)
            correct += (out == lbl).sum().item()
        val_acc = 100.0*correct/len(valid_loader.dataset)
        val_loss /= (len(valid_loader.dataset)/batch_size)
    print(f'\nEpoch: {epoch+1}/{epochs}')
    print(f'Train loss: {train_loss}, Train Accuracy: {train_acc}')
    print(f'Validation loss: {val_loss}, Validation Accuracy: {val_acc}\n')

    scheduler.step(val_loss)

    # write to tensorboard
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)

    if val_acc>best:
        best=val_acc
        torch.save(model,model_name)
        early_stop_count=0
        print('Accuracy Improved, model saved.\n')
    else:
        early_stop_count+=1

    if early_stop_count==ES_patience:
        print('Early Stopping Initiated...')
        print(f'Best Accuracy achieved: {best:.2f}% at epoch:{epoch+1-ES_patience}')
        print(f'Model saved as {model_name}')
        break
    writer.flush()
# writer.close()    

#####NUTRITION BASED CALCULATION
gender = 'female'
weight = 80
height = 183
age = 26 
activity_coef = 1.55

if(gender == 'Male'): 
    bmr = 655.1 + (9.653 * weight) + (1.850 * 183) - (4.676 * age)
else:
    bmr = 66.47 + (13.75 * weight) + (5.003 * 183) - (6.755 * age)
    
calories_required = bmr * activity_coef

protein_required = 0.3 * calories_required
carb_required = 0.55 * calories_required
fats_required = 0.25 * calories_required

nutr_df = pd.read_csv('nutrition_df2.csv', index_col = 'Unnamed: 0')
nutr_df.index = nutr_df.index.set_names(['nutrient'])
nutr_df= nutr_df.reset_index()
nutr_df = nutr_df[nutr_df['nutrient'].isin(['Calories','Protein','Carbohydrates','Fat'])]


