import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from network import TopNet
from FE import FE
from scipy.io import loadmat
from torch import tensor
from plotUtil import Plotter


class TopologyOptimizer:
    def __init__(self, mesh, matProp, bc, nnSettings, desiredVolumeFraction, densityProjection, overrideGPU = True):
        self.FE = FE(mesh, matProp, bc)
        self.device = self.setDevice(overrideGPU)
        self.desiredVolumeFraction = desiredVolumeFraction
        self.Pltr = Plotter()

        self.densityProjection = densityProjection

        inputDim = 25600 # x and y coordn
        self.topNet = TopNet(nnSettings, inputDim).to(self.device)
        self.objective = 0.




    def setDevice(self, overrideGPU):
        if(torch.cuda.is_available() and (overrideGPU == False) ):
            device = torch.device("cuda:0");
            print("GPU enabled")
        else:
            device = torch.device("cpu")
            print("Running on CPU")
        return device

    def projectDensity(self, x):
        if(self.densityProjection['isOn']):
            b = self.densityProjection['sharpness']
            nmr = np.tanh(0.5*b) + torch.tanh(b*(x-0.5));
            x = 0.5*nmr/np.tanh(0.5*b);
        return x;


    def OptimizeDesign(self, maxEpochs, minEpochs):

        # 降维获得参数，暂时从MATLAB中计算数据，直接读取
        eIntopMat = loadmat('eIntopmat.mat')
        eIntopMat = eIntopMat['eIntopMat'].astype(np.float32)
        eIntopMat = torch.from_numpy(eIntopMat)

        eigvalVec = loadmat('eigvalVec54.mat')
        eigvalVec = eigvalVec['ans'].astype(np.float32)



        self.convergenceHistory = {'compliance': [], 'vol': [], 'grayElems': []}
        # 目标最优解eta
        # x = [-0.662926012059972,-0.694469015028937,-0.669321537104286,-0.00820539355665437,-2.42326955223248,-1.68450588019361,-4.36993465975575,-4.37822937949709,0.514233910303176,-0.533586465169250,
        #        -0.0107377981910312,0.0272003363618547,0.0164942515119255,-2.64127543212076,-2.64734087580336,7.10098483044827,-2.24078778942554,-0.470349109200701,-0.615563418324344,-0.0124180184175293,
        #        -1.39860625524035,-2.36003283845079,0.415870446066922,-0.393604274775291,-0.113546787212505,-0.0237455614507609,-0.272029603064573,-0.647065401205189,-0.618282229644643,-0.603732930751384,
        #        1.52202165824954,-1.12922659682827,4.14069469569070,-3.17966801091786,3.00670359550958,-0.0116442291779283,0.0838487364852567,-2.12665130143044,2.67830649416868,-1.00599283505119,
        #        -0.892427497655791,0.302796414638916,-0.573846785729838,3.01217745070431,2.54566800103205,0.0174819074184729,0.328410211076455,-0.842741542974879,0.305088607119025,0.109521307916837,
        #        2.10817309165022,-0.511266382431000,13.7135189286551, 9.51814632460128]
        # x = torch.tensor(x).view(-1, 1).float().to(self.device)


        x = torch.tensor(eigvalVec).view(-1, 1).float().to(self.device)
        x = eIntopMat

        # x = torch.tensor(range(54)).view(-1, 1).float().to(self.device)

        maxEpochs = 500


        learningRate = 0.08
        alphaMax = 100*self.desiredVolumeFraction # alpha is a penal parameter, ensure constarin
        alphaIncrement= 0.1   # 在本例程中，同样使用惩罚法构建优化目标函数以及其对应的loss
        alpha = alphaIncrement # start
        nrmThreshold = 0.01 # for gradient clipping 防止梯度爆炸
        self.optimizer = optim.Adam(self.topNet.parameters(), amsgrad=True,lr=learningRate)

        beta = 0
        ichange = 1






        for epoch in range(maxEpochs):

            self.optimizer.zero_grad()
            # x = self.applySymmetry(self.xy);  # 对称结构特殊处理
            nn_eta = torch.flatten(self.topNet(x))

            nn_ePhi = torch.matmul(eIntopMat.t().to(self.device), nn_eta)  # eta to ePhi

            #  此操作废弃
            # if epoch>200:
            #     low = -0.99
            #     up = 1
            # if 100<epoch<=200 :
            #     low = -0.80
            #     up = 0.80
            # if epoch < 100:
            #     low = -0.1
            #     up = 0.1

            # 超出[-1,1]区间的值被拉回
            nn_ePhi = torch.clamp(nn_ePhi, min=-0.99, max=1)
            nn_rho = 0.5+0.5*nn_ePhi

            # 结构清晰化投影
            if epoch>300:
                nn_rho = self.projectDensity(nn_rho)

            rho_np = nn_rho.cpu().detach().numpy()

            u, Jelem = self.FE.solve(rho_np)

            if(epoch == 0):
                self.obj0 = ( self.FE.Emax*(rho_np**self.FE.penal)*Jelem).sum()

            Jelem = np.array(self.FE.Emax*(rho_np**(2*self.FE.penal))*Jelem).reshape(-1)
            Jelem = torch.tensor(Jelem).view(-1).float().to(self.device)
            objective = torch.sum(torch.div(Jelem,nn_rho**self.FE.penal))/self.obj0  # compliance


            volConstraint = ((torch.mean(nn_rho) / self.desiredVolumeFraction) - 1.0)
            currentVolumeFraction = np.average(rho_np)
            self.objective = objective
            loss = self.objective + alpha * torch.pow(volConstraint, 2)

            alpha = min(alphaMax, alpha + alphaIncrement)
            loss.backward(retain_graph=True)
            # torch.nn.utils.clip_grad_norm_(self.topNet.parameters(), nrmThreshold)
            self.optimizer.step()

            greyElements= sum(1 for rho in rho_np.squeeze() if ((rho > 0.2) & (rho < 0.8)))  # 计算灰色单元数量
            relGreyElements = self.desiredVolumeFraction*greyElements/len(rho_np)
            self.convergenceHistory['compliance'].append(self.objective.item());
            self.convergenceHistory['vol'].append(currentVolumeFraction);
            self.convergenceHistory['grayElems'].append(relGreyElements);
            # self.FE.penal = min(8.0,self.FE.penal + 0.02); # continuation scheme


            if(epoch % 1 == 0):
                titleStr = "Iter {:d} , Obj {:.2F} , vol {:.2F}".format(epoch, self.objective.item()*self.obj0, currentVolumeFraction);
                # self.Pltr.plotDensity(self.xy.detach().cpu().numpy(), rho_np.reshape((self.FE.nelx, self.FE.nely)), titleStr);
                plt.clf()

                plt.imshow(np.reshape(rho_np, (160, 160)), cmap=plt.cm.binary)
                plt.clim(0, 1)
                plt.colorbar(ticks=np.linspace(0,1,11))
                plt.pause(0.05)
                print(titleStr)
                print(loss)
