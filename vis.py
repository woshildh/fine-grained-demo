import net,config,os
import ImageReader
import torch
from torchvision import transforms
import numpy as np

model=net.get_net(classes_num=200,channel_size=2048)

model.load_state_dict(
	torch.load("./weights/cub/resnet50_cub_best_acc.pth"),strict=True)
transform=transforms.Compose([transforms.Resize([224,224]),
	transforms.ToTensor(),
	transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))])
model.eval()
#model.get_max_pos_count("./utils/ori/101.jpg",transform)
#model.get_middle_attention_map("./utils/ori/18.jpg",transform,"./utils/new/res18_middle.jpg")
img_list=os.listdir("./utils/ori/")
for img in img_list:
	classes_num=int(img.split(".")[0])
	#model.get_attention_map("./utils/ori/"+img,transform,classes_num,
	#	"./utils/new/"+img)
	model.get_res("./utils/ori/"+img,transform,classes_num,
		"./utils/new/"+img)	
