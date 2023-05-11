import numpy as np
import random
import torch
import torch.nn as nn
from scipy.io import loadmat

def set_seed(manualSeed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    np.random.seed(manualSeed)
    random.seed(manualSeed)
#%% Neural network

class TopNet(nn.Module):
    def __init__(self, nnSettings, inputDim):
        self.inputDim = inputDim; # x and y coordn of the point
        self.outputDim = 1; # if material/void at the point
        super().__init__();
        self.layers = nn.ModuleList();
        manualSeed = 1234; # NN are seeded manually
        set_seed(manualSeed);
        current_dim = self.inputDim;
        
        for lyr in range(nnSettings['numLayers']): # define the layers
            l = nn.Linear(current_dim, nnSettings['numNeuronsPerLyr']);
            nn.init.xavier_normal_(l.weight);
            nn.init.zeros_(l.bias);
            self.layers.append(l);
            current_dim = nnSettings['numNeuronsPerLyr'];
        self.layers.append(nn.Linear(current_dim, self.outputDim));
        self.bnLayer = nn.ModuleList();
        for lyr in range(nnSettings['numLayers']): # batch norm
            self.bnLayer.append(nn.BatchNorm1d(nnSettings['numNeuronsPerLyr']));

    def forward(self, x):
        m = nn.LeakyReLU();
        ctr = 0;
        for layer in self.layers[:-1]: # forward prop
            x = m(self.bnLayer[ctr](layer(x)));
            ctr += 1;
        # rho = torch.sigmoid(self.layers[-1](x)).view(-1); # grab only the first output
        # rho = 0.01 + (rho-0.5)*2

        #eta的值没有范围 通过ηWη控制

        eta = torch.sigmoid(self.layers[-1](x)).view(-1)
        eta = (eta-0.5)*30

        # eta = [-0.662926012059972,-0.694469015028937,-0.669321537104286,-0.00820539355665437,-2.42326955223248,-1.68450588019361,-4.36993465975575,-4.37822937949709,0.514233910303176,-0.533586465169250,
        #        -0.0107377981910312,0.0272003363618547,0.0164942515119255,-2.64127543212076,-2.64734087580336,7.10098483044827,-2.24078778942554,-0.470349109200701,-0.615563418324344,-0.0124180184175293,
        #        -1.39860625524035,-2.36003283845079,0.415870446066922,-0.393604274775291,-0.113546787212505,-0.0237455614507609,-0.272029603064573,-0.647065401205189,-0.618282229644643,-0.603732930751384,
        #        1.52202165824954,-1.12922659682827,4.14069469569070,-3.17966801091786,3.00670359550958,-0.0116442291779283,0.0838487364852567,-2.12665130143044,2.67830649416868,-1.00599283505119,
        #        -0.892427497655791,0.302796414638916,-0.573846785729838,3.01217745070431,2.54566800103205,0.0174819074184729,0.328410211076455,-0.842741542974879,0.305088607119025,0.109521307916837,
        #        2.10817309165022,-0.511266382431000,13.7135189286551, 9.51814632460128]
        # eta = torch.tensor(eta) # 测试用，最优结构的eta数据
        return  eta;
