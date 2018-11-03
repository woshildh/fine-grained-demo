'''
定义准确率计算和混淆矩阵计算这部分
'''

import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
import time

def get_ctime():
	t=time.strftime("%Y-%m-%d %H:%M:%S")
	return t

class Accuracy(nn.Module):
	def __init__(self):
		super(Accuracy,self).__init__()
		self.total_sample=0
		self.total_correct=0
	def get_acc(self):
		return self.total_correct/(self.total_sample+1e-5)
	def forward(self,logits,labels):
		'''
		params:
			logits:预测的score,shape is [batch_size,classes_num]
			labels:标签,shape is [batch_size], LongTensor
		return:
			acc:float
		'''
		max_value,max_pos=torch.max(logits,dim=1)
		correct_num=torch.eq(max_pos,labels).sum().data[0]
		batch_size=labels.size(0)

		self.total_sample+=batch_size
		self.total_correct+=correct_num

		acc=correct_num/batch_size

		return acc

class AccuracyList(nn.Module):
	def __init__(self,num=1):
		super(AccuracyList,self).__init__()
		self.num=num
		self.total_sample=np.zeros(shape=(num))
		self.total_correct=np.zeros(shape=(num))
	def set_zeros(self):
		self.total_sample=np.zeros(shape=(self.num))
		self.total_correct=np.zeros(shape=(self.num))
	def forward(self,logits,labels):
		'''
		params:
			logits:一个元组,预测的score,shape is [batch_size,classes_num]
			labels:标签,shape is [batch_size], LongTensor
		return:
			acc:float
		'''
		acc_list=[]
		for i,logit in enumerate(logits):
			max_value,max_pos=torch.max(logit,dim=1)
			correct_num=torch.eq(max_pos,labels).sum().data[0]
			batch_size=labels.size(0)
			self.total_sample[i]+=batch_size
			self.total_correct[i]+=correct_num
			acc_list.append(correct_num/batch_size)
		return acc_list
	def get_acc_list(self):
		return list(self.total_correct/(self.total_sample+1e-5))

class LossList(object):
	def __init__(self,num=1):
		self.num=num
		self.steps=0
		self.loss=np.zeros(shape=(num))
	def log(self,loss_list):		
		self.loss+=np.array(loss_list)
		self.steps+=1
	def get_loss_list(self):
		return list(self.loss/(self.steps+1e-5))
	def set_zeros(self):
		self.steps=0
		self.loss=np.zeros(shape=(self.num))		

class Logger(object):
	def __init__(self,csv_path,tb_path):
		self.csv_path=csv_path
		self.tb_path=tb_path
		self.tb_writer=SummaryWriter(log_dir=tb_path)
	def log(self,steps,train_acc_list,train_loss_list,val_acc_list,val_loss_list,
		l2_reg,lr):
		self.__logcsv__(steps,train_acc_list,train_loss_list,val_acc_list,val_loss_list,l2_reg,lr)
		self.__logtb__(steps,train_acc_list,train_loss_list,val_acc_list,val_loss_list,l2_reg,lr)
	def __logcsv__(self,steps,train_acc_list,train_loss_list,val_acc_list,val_loss_list,
		l2_reg,lr):
		with open(self.csv_path,"a",encoding="utf-8") as file:
			for i in range(len(train_acc_list)):
				train_acc_list[i]="{:.4f}".format(train_acc_list[i])
			for i in range(len(train_loss_list)):
				train_loss_list[i]="{:.6f}".format(train_loss_list[i])
			for i in range(len(val_acc_list)):
				val_acc_list[i]="{:.4f}".format(val_acc_list[i])
			for i in range(len(val_loss_list)):
				val_loss_list[i]="{:.6f}".format(val_loss_list[i])
			steps=str(steps)
			lr=str(lr)
			l2_reg="{:.6f}".format(l2_reg)
			t=get_ctime()
			content=[steps]+train_acc_list+train_loss_list+val_acc_list+ \
				val_loss_list+[lr,l2_reg,t]
			content=",".join(content)+"\n"
			file.write(content)
	def __logtb__(self,steps,train_acc_list,train_loss_list,val_acc_list,val_loss_list,
		l2_reg,lr):
		
		self.tb_writer.add_scalar("lr",lr,steps)
		self.tb_writer.add_scalar("l2_reg",l2_reg,steps)

		for i in range(len(train_acc_list)):
			name="Train/Acc/acc{}".format(i)
			self.tb_writer.add_scalar(name,float(train_acc_list[i]),steps)
		for i in range(len(train_loss_list)):
			name="Train/Loss/loss{}".format(i)
			self.tb_writer.add_scalar(name,float(train_loss_list[i]),steps)
		for i in range(len(val_acc_list)):
			name="Val/Acc/acc{}".format(i)
			self.tb_writer.add_scalar(name,float(val_acc_list[i]),steps)
		for i in range(len(val_loss_list)):
			name="Val/Loss/loss{}".format(i)
			self.tb_writer.add_scalar(name,float(val_loss_list[i]),steps)


if __name__=="__main__":
	'''
	log=Logger("./oiuy.csv","./test/t")
	log.log(0,[0.26,0.25987],[1.256,0.254756],[0.025,0.3],[1.23,2.69],12569.321,0.01)
	log.log(1,[0.26,0.25987],[1.256,0.254756],[0.025,0.3],[1.23,2.69],12569.321,0.01)
	log.log(2,[0.26,0.25987],[1.256,0.254756],[0.025,0.3],[1.23,2.69],12569.321,0.01)
	log.log(3,[0.26,0.25987],[1.256,0.254756],[0.025,0.3],[1.23,2.69],12569.321,0.01)
	log.log(4,[0.26,0.25987],[1.256,0.254756],[0.025,0.3],[1.23,2.69],12569.321,0.01)
	log.log(5,[0.26,0.25987],[1.256,0.254756],[0.025,0.3],[1.23,2.69],12569.321,0.01)
	log.log(6,[0.26,0.25987],[1.256,0.254756],[0.025,0.3],[1.23,2.69],12569.321,0.01)
	'''

	pass
