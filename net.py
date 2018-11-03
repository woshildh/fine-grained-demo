import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from model import resnet_image
from PIL import Image
from torchvision.transforms import transforms
import ImageReader,math
import config
import numpy as np,cv2,time,Metrics

def get_features(cnn_weights_path=None):
	features=resnet_image.resnet50()
	if cnn_weights_path is not None:
		features.load_state_dict(torch.load(cnn_weights_path),strict=False)
		print("{} load succeed...".format(cnn_weights_path))
	return features

class Net(nn.Module):
	def __init__(self,classes_num=200,channel_size=2048,drop_rate=0,
		cnn_weights_path=None):
		super(Net,self).__init__()
		self.classes_num=classes_num
		self.features=get_features(cnn_weights_path=cnn_weights_path)
		self.cls=nn.Linear(channel_size,classes_num)
		self.avg=nn.AdaptiveAvgPool2d(output_size=1)
		self.drop=nn.Dropout(p=drop_rate)
		self.criterion=nn.CrossEntropyLoss()
	def forward(self,x1,x2):
		#coaser
		x1=self.features(x1)
		#fine
		x2=self.features(x2)
		b,c,h,w=x2.size()
		#concate
		x1=self.avg(x1).view(b,-1)
		x2=self.avg(x2).view(b,-1)
		res=self.cls(x2)
		return [res,x1,x2]

	def get_valloss(self,logits,labels):
		loss=self.criterion(logits[0],labels)
		res=[loss.data[0]]
		return loss,res
	def get_loss(self,logits,labels):
		loss1=self.criterion(logits[0],labels)
		loss2=self.get_div_loss(logits[1],logits[2])
		loss=loss1+loss2
		res=[loss.data[0],loss1.data[0],loss2.data[0]]
		del(loss1,loss2)
		return loss,res
	def get_div_loss(self,feature1,feature2):
		b,c=feature1.size()
		loss=(feature1-feature2.detach()).pow(2).sum()/b
		return loss
def get_net(classes_num=200,channel_size=2048,cnn_weights_path=None,drop_rate=0):
	model=Net(classes_num=classes_num,channel_size=channel_size,
		drop_rate=drop_rate,cnn_weights_path=cnn_weights_path)
	return model

if __name__=="__main__":
	model=get_net(8)
	inputs1=Variable(torch.randn(1,3,224,224))
	inputs2=Variable(torch.randn(1,3,120,120))
	outputs=model(inputs2,inputs1)
	print(outputs)
