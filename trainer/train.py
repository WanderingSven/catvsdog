import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import tqdm
from torch.nn.parallel import DistributedDataParallel


class Trainer:

    def __init__(self,model,train_dataloader,weight_decay=0.01,lr=0.00001,betas=(0.9, 0.999),ngpu=4,args=None):
        #cuda_condition = torch.cuda.is_available()
        
        #self.device = torch.device('cuda:0' if cuda_condition and ngpu>0 else "cpu")
        self.device = torch.device('cuda', args.local_rank)
        model = model.cuda()
        if ngpu >1:
            cuda_devices = [i for i in range(ngpu)]
            #model = nn.DataParallel(model, device_ids=cuda_devices)
            self.model = DistributedDataParallel(model,device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)
            #self.model = DistributedDataParallel(model,find_unused_parameters=True)

        #self.model = model.to(self.device)

        self.train_data =train_dataloader

        self.optim = Adam(self.model.parameters(),lr=lr,betas=betas, weight_decay=weight_decay)

        self.criterion = nn.CrossEntropyLoss()
    
    def train(self,epoch):
        self.iteration(epoch,self.train_data)

    def dev(self,dev_dataloader):
        self.model.eval()
        sum = 0
        truth_sum = 0
        for input,labels in dev_dataloader:
            sum = sum + input.size(0)
            input = input.to(self.device)
            labels = labels.to(self.device)
            prediction = self.model(input)
            prediction = torch.argmax(prediction,dim=1)

            truth_sum = truth_sum + (prediction == labels ).sum().item()
        print("dev_acc: %f"% (truth_sum/sum))


    
    def iteration(self, epoch, data_loader, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")
        running_loss= 0
        for i,data in data_iter:
            input,label = data
            input = input.to(self.device)
            label = label.to(self.device)            
            prediction = self.model(input)

            probability=F.softmax(prediction,dim=1)
            #print(probability)

            loss = self.criterion(prediction,label)
            if train:
                self.optim.zero_grad()       
                loss.backward()
                self.optim.step()
            running_loss += loss.item()
            if (i+1) % 20 == 0:    # print every 20 mini-batches
                print(' %5d step loss: %.3f' %
                  (i + 1, running_loss / 20))
                running_loss= 0

    def save(self, epoch, file_path=None,tag=None):
        """
        Saving the current  model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + "%s.ep%d" % (tag,epoch)
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)

               





