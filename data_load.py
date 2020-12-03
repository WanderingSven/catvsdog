from torch.utils import data
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import os

transform=transforms.Compose([
                              transforms.Resize((128,128)),
                              transforms.ToTensor(),
                              #transforms.Lambda(lambda x: x.repeat(3,1,1)),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                              ])
def show(input,label):
    #for i,image in enumerate(input):
        plt.imshow(input)
        plt.show()
        print(label)
class dataset(data.Dataset):
    def __init__(self,root,transform=transform,debug=0):
        """
            root: 数据集根文件目录
            transform: 数据预处理
            debug: debug 模式，检测data和lable是否对应
        """
        #super.__init__()
        self.transform = transform
        self.debug = debug
        subpath = [ os.path.join(root,x) for x in os.listdir(root) ]
        self.imgs = []
        for i,path in enumerate(subpath):
            #返回(路径,标签) 元组
            for j in os.listdir(path):
                # 图片路径和它对应的label
                self.imgs.append([os.path.join(path,j),i]) 

    def __getitem__(self,index):
        img_path=self.imgs[index][0]
        from PIL import Image 
        pil_img = Image.open(img_path).convert('RGB')
        if self.debug:
            show(pil_img,self.imgs[index][1])
        if self.transform is not None:
            self.imgs[index][0]= self.transform(pil_img)
        else:
            self.imgs[index][0]= pil_img
        return self.imgs[index][0],self.imgs[index][1]
    def __len__(self):
        return len(self.imgs)


