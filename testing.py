import argparse
import torch
import torch.nn as nn

from nets.vgg import Vgg
from data_load import dataset
from torch.utils.data import DataLoader




def test():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c","--test_dataset",required=True,type=str,help="train dataset for the training")
    parser.add_argument("-s","--save_path",type=str,required=True,help="the path for saving model")


    parser.add_argument("-b","--batch_size",type=int,default=32,help="number of bacth size")
    parser.add_argument("-w","--num_workers",type=int,default=4,help="dataloader worker size")

    parser.add_argument("-n","--ngpu",type=int,default=1,help="the number of gpu")

    args =  parser.parse_args()

    test_dataset = dataset(args.test_dataset,debug=0)

    print("Creating test_dataLoad")
    test_dataload = DataLoader(test_dataset,args.batch_size,num_workers=args.num_workers,shuffle=True)

    print("Loading trained model")
    model = Vgg()
    model_dict = torch.load(args.save_path).module.state_dict()
    model.load_state_dict(model_dict)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.ngpu>0 else "cpu")
    print(device)
    if args.ngpu > 1:
        model = nn.DataParallel(model,device=[i for i in range(args.ngpu)])
    model = model.to(device)
    predict(model,test_dataload,device)


def predict(model,test_dataLoad,device):
        model.eval()
        sum = 0
        truth_sum = 0
        for input,labels in test_dataLoad:
            sum = sum + input.size(0)
            input = input.to(device)
            labels = labels.to(device)
            prediction = model(input)
            #print(prediction)
            prediction = torch.argmax(prediction,dim=1)
            print(prediction)

            truth_sum = truth_sum + (prediction == labels ).sum().item()
        print("dev_acc: %f"% (truth_sum/sum))


if __name__ == "__main__":
    test()




