import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
import gzip
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
#from skimage.transform import resize
from torchvision.ops import distance_box_iou_loss
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision.transforms.v2.functional import resize
from torchvision.transforms import InterpolationMode
def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object
    
def save_zipped_pickle(obj, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, 2)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_factor):
        super(ConvBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(p=drop_factor),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(p=drop_factor)
        )
        
    def forward(self, x):
        return self.layers(x)
def IoULoss(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = torch.sum(pred*target)
    union = torch.sum(pred)+torch.sum(target)-intersection
    return intersection/union

def jaccard_similarity(y_train, y_test):
    y_train_f = y_train.view(-1)
    y_test_f = y_test.view(-1)
    intersection = torch.sum((y_train_f * y_test_f))
    union = torch.sum(y_train_f) + torch.sum(y_test_f) - intersection
    return intersection / union

def jaccard_loss(y_true, y_pred, smooth=100):
    mse = torch.nn.MSELoss()
    intersection = torch.sum(torch.abs(y_true * y_pred))
    sum_ = torch.sum(torch.abs(y_true) + torch.abs(y_pred))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    #return- jac * smooth
    return -jaccard_similarity(y_true, y_pred)

def dice_loss(pred,tar):
    pred =pred.view(-1)
    tar=tar.view(-1)
    epsilon=1e-6
    intersection=(pred*tar).sum()
    sums = torch.sum(pred*tar)-torch.sum(tar*tar)

    dice = (2*intersection+epsilon)/(sums+epsilon)
    return 1-dice
class UNET(nn.Module):
    def __init__(self,in_channels,out_channels,base_features, drop_factor,device):
        super(UNET,self).__init__()
        kern_size=2
        self.device=device

        self.encode1 = ConvBlock(in_channels,base_features, drop_factor).to(self.device)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=kern_size,stride=2)
        
        self.encode2 = ConvBlock(base_features,2*base_features,drop_factor=drop_factor).to(self.device)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=kern_size,stride=2)
        
        self.encode3 = ConvBlock(2*base_features, 4*base_features,drop_factor = drop_factor).to(self.device)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=kern_size,stride=2)
        
        self.encode4 = ConvBlock(4*base_features, 8*base_features,drop_factor = drop_factor).to(self.device)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=kern_size,stride=2)
        
        self.bottleneck = ConvBlock(8*base_features,16*base_features,drop_factor=0)
        
        self.upconv1 = torch.nn.ConvTranspose2d(16*base_features, 8*base_features, kernel_size=kern_size,stride=2)
        self.decode1 = ConvBlock(16*base_features,8*base_features, drop_factor=drop_factor).to(self.device)
        
        self.upconv2 = torch.nn.ConvTranspose2d(8*base_features, 4*base_features, kernel_size=kern_size,stride=2)
        self.decode2 = ConvBlock(8*base_features,4*base_features, drop_factor=drop_factor).to(self.device)
        
        self.upconv3 = torch.nn.ConvTranspose2d(4*base_features, 2*base_features, kernel_size=kern_size,stride=2)
        self.decode3 = ConvBlock(4*base_features,2*base_features, drop_factor=drop_factor).to(self.device)
        
        self.upconv4 = torch.nn.ConvTranspose2d(2*base_features, 1*base_features, kernel_size=kern_size,stride=2)
        self.decode4 = ConvBlock(2*base_features,1*base_features, drop_factor=drop_factor).to(self.device)
        
        self.out = torch.nn.Conv2d(in_channels=base_features,out_channels=out_channels, kernel_size=1)
        self.fin = torch.nn.Sigmoid()
        self.to(self.device)
    def forward(self,x):
        x.to(torch.float32).to(self.device)
        encoded1 = self.encode1(x)
        encoded2 = self.encode2(self.pool1(encoded1))
        encoded3 = self.encode3(self.pool2(encoded2))
        encoded4 = self.encode4(self.pool3(encoded3))
        
        bottled = self.bottleneck(self.pool4(encoded4))
        
        decoded1 = self.upconv1(bottled)
        decoded1 = torch.cat((decoded1,encoded4),dim=1)
        decoded1 = self.decode1(decoded1)
        
        decoded2 = self.upconv2(decoded1)
        decoded2 = torch.cat((decoded2,encoded3),dim=1)
        decoded2 = self.decode2(decoded2)
        
        decoded3 = self.upconv3(decoded2)
        decoded3= torch.cat((decoded3,encoded2),dim=1)
        decoded3 = self.decode3(decoded3)
        
        decoded4 = self.upconv4(decoded3)
        decoded4= torch.cat((decoded4,encoded1),dim=1)
        decoded4 = self.decode4(decoded4)
        return self.fin(self.out(decoded4))
    def train_step(self, batch,loss_func,optim):
        inp,tar = batch
        mse = nn.MSELoss()
        #dice_loss = DiceLoss()
        inp = inp.to(self.device)
        tar = tar.to(self.device)
        pred = self.forward(inp)
        optim.zero_grad()
        loss = loss_func(pred,tar)
        #loss = loss_func(pred,tar)
        #loss = dice_loss(pred,tar)+ IoULoss(pred, tar)
        #loss += 1/75*dice_loss(pred, tar)+1/50*mse(pred,tar)
        #loss+= 1/25*mse(pred,tar)
        #loss +=1/10*dice_loss(pred,tar)
        #loss += jaccard_loss(tar,pred)
        loss.backward()
        optim.step()
        return loss
    
    def train_model(self, train_loader, val_loader=None, num_epochs=100):
        optim = torch.optim.Adam(self.parameters(), lr=0.1, weight_decay=1e-6)
        scheduler = lr_scheduler.StepLR(optim, step_size=30, gamma=0.1)
        reduce_lr_scheduler = ReduceLROnPlateau(optim, mode='min', patience=5, factor=0.1, verbose=True)
        
        loss_func = jaccard_loss
        val_hist = []
        
        for epoch in range(num_epochs):
            self.train()
            total_train_loss = 0.0
            num_train_batches = len(train_loader)
        
            train_progress_bar = tqdm(enumerate(train_loader), total=num_train_batches, desc=f'Epoch {epoch + 1}/{num_epochs} (Training)', unit='batch')
        
            for batch_idx, batch in train_progress_bar:
                loss = self.train_step(batch, loss_func, optim)
                total_train_loss += loss.item()
                train_progress_bar.set_postfix(train_loss=loss.item())
        
            avg_train_loss = total_train_loss / num_train_batches
        
            self.eval()
            
            if val_loader is not None:
                # Validation
                self.eval()
                total_val_loss = 0.0
                num_val_batches = len(val_loader)

                val_progress_bar = tqdm(enumerate(val_loader), total=num_val_batches, desc=f'Epoch {epoch + 1}/{num_epochs} (Validation)', unit='batch')

                with torch.no_grad():
                    for batch_idx, batch in val_progress_bar:
                        inp, tar = batch
                        inp = inp.to(self.device)
                        tar = tar.to(self.device)
                        inp = inp#+torch.randn_like(inp)
                        pred = self.forward(inp)
                        val_loss = loss_func(pred, tar)
                        total_val_loss += val_loss.item()
                        val_progress_bar.set_postfix(val_loss=val_loss.item())

                avg_val_loss = total_val_loss / num_val_batches
                reduce_lr_scheduler.step(avg_val_loss) 
                val_hist.append(1+avg_val_loss)
                print(f'Epoch {epoch + 1}/{num_epochs}, Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}')
        
        print('Training complete.')
        return val_hist
    
def handle_data_train(data):
    inps = []
    targs = []
    vid = data['video']
    frames = data['frames']
    label = data['label']
    new_img_shape = (128, 128)

    for frame in range(len(frames)):
        # Extract the frame from video and label
        pic = vid[:, :, frames[frame]]
        lab = label[:, :, frames[frame]] * 1

        # Resize the frame
        pic_new = resize(torch.tensor(pic).unsqueeze(0).to(torch.float32),size= new_img_shape,  interpolation=InterpolationMode.BILINEAR, antialias=True)
        lab_new = resize(torch.tensor(lab).unsqueeze(0),size= new_img_shape,  interpolation=InterpolationMode.BILINEAR, antialias=True)
        lab_new = (lab_new>0)*1.0
        # Append the resized frames to the lists
        inps.append(pic_new)
        targs.append(lab_new)

    return torch.stack(inps), torch.stack(targs)

def handle_data_test(data):
    inps = []
    vid = data['video']
    new_img_shape = (128, 128)
    
    for i in range(vid.shape[2]):
        inps.append(resize(torch.tensor(vid[:, :, i]).unsqueeze(0), new_img_shape, interpolation=InterpolationMode.BILINEAR, antialias=True).to(torch.float32))
    
    return torch.stack(inps)

def create_train_set_test_set(train_data, test_size=0.25, batch_size=32, shuffle=True):
    X_train, y_train = [], []
    for entry in train_data:
        inps, targs = handle_data_train(entry)
        X_train.append(inps)
        y_train.append(targs)

    X_train = torch.cat(X_train,dim=0).to(torch.float32)
    y_train = torch.cat(y_train,dim=0).to(torch.float32)

    train_dataset = TensorDataset(X_train, y_train)

    X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X_train, y_train, test_size=test_size,
                                                                           shuffle=shuffle, random_state=42)
    
    #X_train_data = X_train
    #y_train_data = y_train
    train_loader = DataLoader(TensorDataset(X_train_data, y_train_data), batch_size=batch_size, shuffle=shuffle)

    test_loader = DataLoader(TensorDataset(X_test_data, y_test_data), batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader
            
train_data = load_zipped_pickle("task3/train.pkl")
test_data = load_zipped_pickle("task3/test.pkl")
samples = load_zipped_pickle("task3/sample.pkl")

train,test = create_train_set_test_set(train_data)

#train= create_train_set_test_set(train_data)

device="cuda"
unet = UNET(in_channels=1,out_channels=1,base_features=2,drop_factor=0,device=device)##35 epochs with p=0.2 works more or less
val_hist = unet.train_model(train_loader=train,val_loader = test,num_epochs=200)
unet.eval()
predictions =[]
for data in test_data:
    prediction = np.array(np.zeros_like(data['video']), dtype=bool)
    X_in = handle_data_test(data)
    print(len(X_in))
    for i, frame in enumerate(X_in):
        #plt.imshow(frame.squeeze(0).cpu().numpy())
        #plt.show()  
        pred = unet(frame.to(device).unsqueeze(0)).squeeze(0)
        pred = pred > 0.4
        
        pred = pred.to(torch.bool)
        plt.imshow(pred[0,:,:].cpu().detach())
        plt.show()
        
        prediction[:, :, i] = resize(pred, size=prediction.shape[:2], interpolation=InterpolationMode.BILINEAR, antialias=True).detach().cpu().numpy()
    
    predictions.append({
        'name': data['name'],
        'prediction': prediction
    })
save_zipped_pickle(predictions, 'my_predictions.pkl')
plt.loglog(val_hist)
