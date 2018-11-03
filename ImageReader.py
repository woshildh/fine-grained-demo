import torch,os,config
from torchvision import transforms
from  torchvision.datasets import ImageFolder,CIFAR10,CIFAR100
from torch.utils.data import DataLoader
from torchvision.utils import save_image,make_grid
import torchvision

t1=	transforms.Compose([
	transforms.Resize(130),transforms.RandomCrop(120),
	transforms.ToTensor(),
	transforms.Normalize(config.data_mean,config.data_std)])
t2=transforms.Compose([
	transforms.Resize(256),transforms.transforms.RandomCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(config.data_mean,config.data_std)])
train_transforms_list=[t1,t2]
t1=	transforms.Compose([
	transforms.Resize(130),transforms.CenterCrop(120),
	transforms.ToTensor(),
	transforms.Normalize(config.data_mean,config.data_std)])
t2=transforms.Compose([
	transforms.Resize(256),transforms.transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(config.data_mean,config.data_std)])
val_transforms_list=[t1,t2]


class MultiScaleImageFolder(ImageFolder):
	def __init__(self,root,loader=None,transforms_list=None):
		super(MultiScaleImageFolder,self).__init__(root=root)
		self.num=len(transforms_list)
		self.transforms_list=transforms_list
	def __getitem__(self,index):
		path,target=self.samples[index]
		sample=self.loader(path)
		res_list=[]
		for i in range(self.num):
			res_list.append(self.transforms_list[i](sample))
		return res_list,target
	def __len__(self):
		return len(self.samples)

def get_loader(mode,folder_path):
	if mode=="train":
		dataset=MultiScaleImageFolder(folder_path,transforms_list=train_transforms_list)
		data_loader=DataLoader(dataset,batch_size=config.batch_size,
			shuffle=True,num_workers=config.num_workers)
	elif mode=="validate":
		dataset=MultiScaleImageFolder(folder_path,transforms_list=val_transforms_list)
		data_loader=DataLoader(dataset,batch_size=config.batch_size,
			shuffle=False,num_workers=config.num_workers)
	return data_loader

if __name__=="__main__":
	'''
	tran_list=[
		transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor()]),
		transforms.Compose([transforms.Resize([300,300]),transforms.ToTensor()])
		]
	d=MultiScaleImageFolder(root="../data/cub/CUB200/images/train/",
		transforms_list=tran_list)
	l=DataLoader(d,batch_size=16)
	for (x1,x2),y in l:
		print(x1.size())
		print(x2.size())
		print(y.size())
		break
	'''
	root="../data/cub/CUB200/images/train/"
	loader=get_loader("train",root)
	for (x1,x2),y in loader:
		print(x1.size(),x2.size(),y.size())

