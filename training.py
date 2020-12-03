import torch.nn as nn
import argparse
from nets.vgg import Vgg
from nets.resnet import ResNet
from trainer.train import Trainer
from torchvision import models

from data_load import dataset
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.utils.data.distributed

def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c","--train_dataset",required=True,type=str,help="train dataset for the training")
    parser.add_argument("-d","--dev_dataset",type=str,default=None,help="dev dataset path")
    parser.add_argument("-o","--out_path",type=str,help="the path for saving model")


    parser.add_argument("-b","--batch_size",type=int,default=16,help="number of bacth size")
    parser.add_argument("-e","--epochs",type=int,default=50,help="number of epochs")
    parser.add_argument("-w","--num_workers",type=int,default=4,help="dataloader worker size")

    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    
    parser.add_argument("-n","--ngpu",default=1,type=int,help="the number of used gpu")

    parser.add_argument("--debug",default=0,type=bool,help="debug mode if 1")

    # distributed training
    parser.add_argument('--rank', default=0,
                     help='rank of current process')
    # parser.add_argument('--word_size', default=2,
    #                 help="word size")
    # parser.add_argument('--init_method', default='tcp://127.0.0.1:43456',
    #                 help="init-method")
    parser.add_argument("--local_rank",default=0, type=int)
   


    args = parser.parse_args()
    print("local_rank %d" %args.local_rank)
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl',init_method='env://')
    print("distributed init")
    #dist.init_process_group(backend='nccl', init_method=args.init_method, rank=args.rank, world_size=args.word_size)

    print("Loading Train Dataset", args.train_dataset)
    train_dataset = dataset(args.train_dataset,debug=0)
    if args.dev_dataset is not None:
        dev_dataset = dataset(args.dev_dataset)

    if args.ngpu >1 :
        args.batch_size = args.batch_size * args.ngpu
    print("Creating Dataloader")

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataload= DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,sampler=train_sampler)
    if args.dev_dataset is not None:
        dev_dataload = DataLoader(dev_dataset, batch_size=args.batch_size, num_workers=args.num_workers,sampler=train_sampler)

    print("Building  model")
    # net=models.vgg11()
    # net.classifier._modules['6'] = nn.Linear(4096, 2)
    # model = net
    # model = Vgg()
    model = ResNet()

    print("Creating Trainer")
    trainer = Trainer(model,train_dataload,lr=args.lr,weight_decay=args.adam_weight_decay,ngpu=args.ngpu,args=args)
    
    #tag = "vgg"
    tag = "Resnet"
    print("Training Start")
    for epoch in range(args.epochs):
        trainer.train(epoch)
        if args.dev_dataset is not None:
            trainer.dev(dev_dataload)
        if args.out_path is not None:
            trainer.save(epoch,args.out_path,tag=tag)

if __name__ == "__main__":
    train()
    
    






