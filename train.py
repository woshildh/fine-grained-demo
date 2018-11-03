import torch,os
from torchvision import transforms
import config,ImageReader,Metrics,net
from tensorboardX import SummaryWriter
from torch.optim import SGD
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.backends.cudnn as cudnn

#设置使用gpu的id
if config.use_cuda:
	os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu_id

def train():
	'''
	进行训练
	'''
	#定义记录部分
	log=Metrics.Logger(config.csv_path,config.tb_path)
	#定义模型
	if config.start_epoch==1:
		model=net.get_net(classes_num=config.classes_num,
			channel_size=config.channel_size,
			cnn_weights_path=config.cnn_weights_path,drop_rate=config.drop_rate)
	else:
		model=net.get_net(classes_num=config.classes_num,
			channel_size=config.channel_size,drop_rate=config.drop_rate)

	print("model load succeed")

	if config.use_cuda:
		model=model.cuda() #将model转到cuda上
	if config.use_parallel:
		model=nn.DataParallel(model,device_ids=config.device_ids)
		cudnn.benchmark=True
	if config.start_epoch!=1:
		all_weights_path=config.save_weights_path.format(config.start_epoch-1)
		model.load_state_dict(torch.load(all_weights_path))
		print("{} load succeed".format(all_weights_path))
	
	#加载数据集
	train_folder=config.train_img_path
	validate_folder=config.validate_img_path
	train_loader=ImageReader.get_loader("train",train_folder)
	validate_loader=ImageReader.get_loader("validate",validate_folder)

	#定义优化器和学习率调度器
	optimizer=SGD(params=model.parameters(),lr=config.start_lr,
		momentum=0.9,weight_decay=config.weight_decay)
	scheduler=optim.lr_scheduler.MultiStepLR(optimizer,milestones=config.stones,
		gamma=0.1)

	#定义评估函数
	accuracy=Metrics.AccuracyList(num=1)
	l_train=Metrics.LossList(1)
	l_val=Metrics.LossList(1)

	#定义最好的准确率
	best_acc=0
	for i in range(config.start_epoch,config.start_epoch+config.num_epoch):
		#分配学习率
		scheduler.step(epoch=i)
		lr=scheduler.get_lr()[0]

		print("{} epoch start , lr is {}".format(i,lr))

		#开始训练这一轮
		model.train()
		accuracy.set_zeros()
		l_train.set_zeros()
		train_step=0
		for (x1,x2),y in train_loader:
			x1=Variable(x1)
			x2=Variable(x2)
			y=Variable(y)
			if config.use_cuda:
				x1=x1.cuda()
				x2=x2.cuda()
				y=y.cuda(async=True)
			optimizer.zero_grad() #清空梯度值
			y_=model(x1,x2) #求y
			#求这一步的损失值和准确率
			step_loss,loss_list=model.get_loss(y_,y)
			step_acc=accuracy(y_,y)
			l_train.log(loss_list)

			#更新梯度值
			step_loss.backward()
			optimizer.step()
			
			train_step+=1  #训练步数+1

			#输出这一步的记录
			print("{} epoch,{} step,step loss is {:.6f},step acc is {:.4f}".format(
				i,train_step,loss_list[0],max(step_acc)))
			del(step_loss,x1,x2,y,y_)
		#求这一轮训练情况
		train_acc_list=accuracy.get_acc_list()
		train_loss_list=l_train.get_loss_list()

		#保存模型
		weights_name=config.save_weights_path.format(i)
		torch.save(model.state_dict(),weights_name)
		del_weights_name=config.save_weights_path.format(i-1)
		if os.path.exists(del_weights_name):
			os.remove(del_weights_name)
		print("{} save,{} delete".format(weights_name,del_weights_name))

		#开始验证步骤
		model.eval()
		accuracy.set_zeros()  #将accuracy中total_sample和total_correct清0
		l_val.set_zeros()
		for (x1,x2),y in validate_loader:
			x1=Variable(x1,requires_grad=False)
			x2=Variable(x2,requires_grad=False)
			y=Variable(y,requires_grad=False)
			if config.use_cuda:
				x1=x1.cuda()
				x2=x2.cuda()
				y=y.cuda(async=True)
			y_=model(x1,x2)
			_,loss_list=model.get_valloss(y_,y)
			accuracy(y_,y)
			l_val.log(loss_list)
			del(x1,x2,y,y_,_)
		val_acc_list=accuracy.get_acc_list()
		val_loss_list=l_val.get_loss_list()
		print("validate end,log start")
		
		#保存最佳的模型
		if best_acc<max(val_acc_list):
			weights_name=config.save_weights_path.format("best_acc")
			torch.save(model.state_dict(),weights_name)
			best_acc=max(val_acc_list)

		#求model的正则化项
		l2_reg=0.0
		for param in model.parameters():
			l2_reg += torch.norm(param).data[0]
		#开始记录
		log.log(i,train_acc_list,train_loss_list,val_acc_list,val_loss_list,
			l2_reg,lr)
		print("log end ...")


		print("{} epoch end, train loss is {},train acc is {},val loss is {},val acc is {},weight l2 norm is {}".format(i,train_loss_list[0],max(train_acc_list),val_loss_list[0],max(val_acc_list),l2_reg))
	del(model)
	print("{} train end,best_acc is {}...".
		format(config.dataset,best_acc))

if __name__=="__main__":
	train()

