import torch, sys
import torch.nn as nn
import numpy as np
from tqdm import tqdm, trange
from cnn_utils import CRPSLoss


class ConvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, activation=nn.LeakyReLU(0.2, inplace=True), groups=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=groups)
        self.activation = activation
        self.bnorm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(0.25)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.bnorm(x)
        x = self.dropout(x)
        return x
    
    
class ConvPool(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, activation=nn.LeakyReLU(0.2, inplace=True), groups=1):
        super(ConvPool, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels, kernel_size, stride, padding, bias, activation, groups)
        self.pool = nn.MaxPool2d(kernel_size=2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x
    
    
class MLP4coords(nn.Module):
        
        def __init__(self, in_features, out_features, hidden_features, bias=True, activation=nn.LeakyReLU(0.2, inplace=True)):
            super(MLP4coords, self).__init__()
            self.fc1 = nn.Linear(in_features, hidden_features, bias)
            self.fc2 = nn.Linear(hidden_features, hidden_features, bias)
            self.fc3 = nn.Linear(hidden_features, out_features, bias)
            self.activation = activation
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.activation(x)
            x = self.fc2(x)
            x = self.activation(x)
            x = self.fc3(x)
            return x
    
    
class CNN4PP(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, activation=nn.LeakyReLU(0.2, inplace=True),
                 deterministic=False):
        super(CNN4PP, self).__init__()
        
        self.conv1 = ConvPool(in_channels, out_channels, kernel_size, stride, padding, bias, activation, groups=1)
        self.conv2 = ConvPool(out_channels, out_channels*2, kernel_size, stride, padding, bias, activation, groups=1)
        self.conv3 = ConvPool(out_channels*2, out_channels*4, kernel_size, stride, padding, bias, activation, groups=1)
        self.conv4 = ConvPool(out_channels*4, 1, kernel_size, stride, padding, bias, activation, groups=1)
        self.input_mlp = MLP4coords(3, 4, 8)
        
        
        self.s1 = (in_channels, 241, 241)
        self.s2 = conv_output_shape(*self.s1, out_channels, kernel_size, padding, stride, dil=1)
        self.s3 = conv_output_shape(*self.s2, out_channels*2, kernel_size, padding, stride, dil=1)
        self.s4 = conv_output_shape(*self.s3, out_channels*4, kernel_size, padding, stride, dil=1)
        self.s5 = conv_output_shape(*self.s4, 1, kernel_size, padding, stride, dil=1)
        if deterministic:
            self.final_mlp = MLP4coords(np.prod(self.s5)+4, 2, hidden_features=8)
        else:
            self.final_mlp = MLP4coords(np.prod(self.s5)+4, 4, hidden_features=8)
    
    def forward(self, x, y):
        y = self.input_mlp(y)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.final_mlp(torch.cat((nn.Flatten()(x), y), dim=1))
        return x

    def print_(self):
        #print(self)
        print("Shapes:\n")
        print(self.s1)
        print(self.s2)
        print(self.s3)
        print(self.s4)
        print(self.s5)            


def CNN_train(model, subset_train_loader, train_loader, val_loader, optimizer, num_epochs, scheduler=None, deterministic=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    criterion = nn.MSELoss() if deterministic else CRPSLoss

    train_losses = []
    val_losses = []
    if not deterministic:
        val_indiv_losses = []
        train_indiv_losses = []
    model = model.to(device)

    # train for 2 epochs to avoid problems with weights init
    if subset_train_loader is not None:
        with trange(1, 3, desc=f'Early training (10% train data)', unit='epoch') as t:
            for epoch in t:
                model.train()
                with tqdm(subset_train_loader, desc=f'Early Train epoch {epoch}',
                unit='batch', leave=False) as t1:
                    for batch_idx, (fields, coords, targets) in enumerate(t1):
                        fields, coords, targets = fields.float().to(device), coords.float().to(device), targets.float().to(device)
                        
                        optimizer.zero_grad()
                        outputs = model(fields, coords)

                        if not deterministic:
                            loss, _ = criterion(outputs, targets)
                        else:
                            loss = criterion(outputs, targets)
                        
                        loss.backward()
    
    with trange(1, num_epochs + 1, desc=f'Training', unit='epoch') as t:
        
        for epoch in t:
            model.train()
            running_loss = 0.0
            if not deterministic:
                train_indiv_loss = np.array([0.0]*4)
            
            with tqdm(train_loader, desc=f'Train epoch {epoch}',
              unit='batch', leave=False) as t1:
                
                for batch_idx, (fields, coords, targets) in enumerate(t1):
                    fields, coords, targets = fields.float().to(device), coords.float().to(device), targets.float().to(device)
                    
                    optimizer.zero_grad()
                    
                    outputs = model(fields, coords)
                    if not deterministic:
                        loss, indiv_loss = criterion(outputs, targets)
                        train_indiv_loss += indiv_loss.mean(axis=0)
                    else:
                        loss = criterion(outputs, targets)
                    
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.detach().item()
            
            epoch_loss = running_loss / len(train_loader)
            if not deterministic:
                train_indiv_loss /= len(train_loader)
                train_indiv_losses.append(train_indiv_loss)
            train_losses.append(epoch_loss)
            
            # Calculate loss on validation dataset
            val_loss = 0.0
            if not deterministic:
                val_indiv_loss = np.array([0.0]*4)
            model.eval()
            with torch.no_grad():
                with tqdm(val_loader, desc=f'Val. epoch {epoch}',
                unit='batch', leave=False) as t2:
                    for batch_idx, (fields, coords, targets) in enumerate(t2):
                        fields, coords, targets = fields.float().to(device), coords.float().to(device), targets.float().to(device)
                        
                        val_outputs = model(fields, coords)
                        if not deterministic:
                            loss, indiv_loss = criterion(val_outputs, targets)
                            val_indiv_loss += indiv_loss.mean(axis=0)
                        else:
                            loss = criterion(val_outputs, targets)
                        val_loss += loss.item()
            
            val_loss /= len(val_loader)
            if not deterministic:
                val_indiv_loss /= len(val_loader)
                val_indiv_losses.append(val_indiv_loss)
            val_losses.append(val_loss)
            
            # Update scheduler if provided
            if scheduler is not None:
                scheduler.step()

    return (np.array(train_losses), np.array(val_losses)) if deterministic else (np.array(train_indiv_losses), np.array(val_indiv_losses))
    
"""class CNN4PP_Indep(nn.Module):
    
    # Designed to separate the two paths (wind/pres) -> each have its own set of weights in filters
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, activation=nn.LeakyReLU(0.2, inplace=True)):
        super(CNN4PP_Indep, self).__init__()
        if not in_channels%2==0 or not out_channels%2==0:
            raise ValueError("in_channels and out_channels must be even, you passed {} and {}".format(in_channels, out_channels))
        
        self.conv1 = ConvPool(in_channels, out_channels, kernel_size, stride, padding, bias, activation, groups=2)
        self.conv2 = ConvPool(in_channels, out_channels*2, kernel_size, stride, padding, bias, activation, groups=2)
        self.conv3 = ConvPool(in_channels, out_channels*4, kernel_size, stride, padding, bias, activation, groups=2)
        self.conv4 = ConvPool(in_channels, out_channels*8, kernel_size, stride, padding, bias, activation, groups=2)
        
        self.final_conv = ConvBlock(out_channels*8, 1, kernel_size, stride, padding, bias, activation)
        self.mlp = MLP4coords(3, 4, 64)
        
    def forward(self, x):
        x1, x2 = x[:, 0, :, :], x[:, 1, :, :]
        x1, x2 = self.path1(x1), self.path2(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.final_conv(x)"""
        
        
def conv_output_shape(cin, hin, win, cout, k_size, pad, stride, dil, pool=True):
    
    if not isinstance(k_size, list):
        k_size = [k_size, k_size]
    if not isinstance(pad, list):
        pad = [pad, pad]
    if not isinstance(dil, list):
        dil = [dil, dil]
    if not isinstance(stride, list):
        stride = [stride, stride]
        
    hout = int(np.floor((hin + 2*pad[0] - dil[0]*(k_size[0]-1) - 1)/stride[0] + 1))
    wout = int(np.floor((win + 2*pad[1] - dil[1]*(k_size[1]-1) - 1)/stride[1] + 1))
    
    hout = int(np.floor((hout - 2)/2)+1)
    wout = int(np.floor((wout - 2)/2)+1)
    
    return (cout, hout, wout)
