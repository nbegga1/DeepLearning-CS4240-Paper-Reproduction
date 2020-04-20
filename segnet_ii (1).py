#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# functions of Maddern's method and Alvarez's mathod
def invariant_Maddern(img,alpha):
  ii_image = (0.5 + torch.log(img[1, :, :]) - alpha * torch.log(img[2, :, :]) - (1 - alpha) * torch.log(img[0, :, :]) )
  return ii_image

def log_approx(x):
  alpha = 5000
  log_x = alpha*(pow(x,1/alpha)-1)
  return log_x

def invariant_Alvarez(img,theta):
  ii_image = 0.5 + math.cos(theta)*log_approx(img[0, :, :]/(img[2, :, :])+1e-10) + math.sin(theta)*log_approx(img[1, :, :]/(img[2, :, :])+1e-10)
  return ii_image


# In[180]:


# Specify transforms using torchvision.transforms as transforms
# Process images using Maddern's method
# The one-channel gray image is transformed to have three same channels
res_ratio = 0.1 # set image size
alpha = 0.48

transformationsimg_Mad = transforms.Compose([
    transforms.Resize((int(res_ratio*720),int(res_ratio*960)),interpolation=Image.NEAREST), #set resolution with nearest neighbor interpolation
    transforms.ToTensor(),# Normalize data to be of values [0-1]
    transforms.Lambda(lambda img: invariant_Maddern(img,alpha)),
    transforms.Lambda(lambda img: img.repeat(3,1,1))
])

train = CamVid('/content/drive/My Drive/','train',transformationsimg_Mad, transformationslabel)
val = CamVid('/content/drive/My Drive/','val',transformationsimg_Mad, transformationslabel)
test = CamVid('/content/drive/My Drive/','test',transformationsimg_Mad, transformationslabel)

train_dataloader = data.DataLoader(train, batch_size= 4, shuffle = True, num_workers=4)
val_dataloader = data.DataLoader(val, batch_size= 4, shuffle = True, num_workers=4)
test_dataloader = data.DataLoader(test, batch_size= 4, shuffle = True, num_workers=4)

for index,[img,label] in enumerate(train_dataloader):
    print('image size:',img.size())
    print('label size:',label.size())
    mask_encoded = [label_to_rgb(label[x,:,:].numpy(), color_encoding) for x in range(label.shape[0])]
    for i in range(0,4):
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(img[i].permute(1,2,0))
        plt.subplot(1,2,2)
        plt.imshow(mask_encoded[i])
    break
   


# In[182]:


# Specify transforms using torchvision.transforms as transforms
# Alvarez's method
res_ratio = 0.1 # set image size
theta = 135

transformationsimg_Alvarez = transforms.Compose([
    transforms.Resize((int(res_ratio*720),int(res_ratio*960)),interpolation=Image.NEAREST), #set resolution with nearest neighbor interpolation
    transforms.ToTensor(),# Normalize data to be of values [0-1]
    transforms.Lambda(lambda img: invariant_Alvarez(img,theta)),
    transforms.Lambda(lambda img: img.repeat(3,1,1)), 
])

transformationslabel = transforms.Compose([
    transforms.Resize((int(res_ratio*720),int(res_ratio*960)),interpolation=Image.NEAREST), #set resolution with nearest neighbor interpolation
    transforms.ToTensor() # Normalize data to be of values [0-1]
])

train = CamVid('/content/drive/My Drive/','train',transformationsimg_Alvarez, transformationslabel)
val = CamVid('/content/drive/My Drive/','val',transformationsimg_Alvarez, transformationslabel)
test = CamVid('/content/drive/My Drive/','test',transformationsimg_Alvarez, transformationslabel)

train_dataloader = data.DataLoader(train, batch_size= 4, shuffle = True, num_workers=4)
val_dataloader = data.DataLoader(val, batch_size= 4, shuffle = True, num_workers=4)
test_dataloader = data.DataLoader(test, batch_size= 4, shuffle = True, num_workers=4)

for index,[img,label] in enumerate(train_dataloader):
    print(img.size())
    print(label.size())
    print(img.type())
    mask_encoded = [label_to_rgb(label[x,:,:].numpy(), color_encoding) for x in range(label.shape[0])]
    for i in range(0,4):
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(img[i].permute(1,2,0))
        plt.subplot(1,2,2)
        plt.imshow(mask_encoded[i])
    break
  

