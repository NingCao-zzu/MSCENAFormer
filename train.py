import os
import argparse
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils import AverageMeter
from datasets.loader import PairLoader
from utils.CR import ContrastLoss_res

from torchvision.models import vgg16
from models import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='mscenaformer', type=str, help='model name')
parser.add_argument('--num_workers', default=0, type=int, help='number of workers')
parser.add_argument('--no_autocast', action='store_false', default=False, help='disable autocast')
parser.add_argument('--save_dir', default='./saved_models_ntire1_b2/', type=str, help='path to models saving')
parser.add_argument('--log_dir', default='./logs/', type=str, help='path to logs')
parser.add_argument('--data_dir', default='../data/', type=str, help='path to dataset')
parser.add_argument('--dataset', default='Ntire2021', type=str, help='dataset name')
parser.add_argument('--exp', default='', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0', type=str, help='GPUs used for training')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def train(train_loader, network, criterion, optimizer, scaler):
	# 追踪loss的平均值
	losses = AverageMeter()
	# 清空CUDA缓存。这可以释放GPU上的一些内存，有助于防止在训练过程中内存占用过高。
	torch.cuda.empty_cache()
	# 将神经网络切换到训练模式。
	network.train()
	# 将输入数据（source图像和target图像）移动到GPU上。
	for batch in train_loader:
		source_img = batch['source'].cuda()
		target_img = batch['target'].cuda()

		# 使用混合精度训练中的上下文管理器 autocast。
		# args.no_autocast 控制是否禁用 autocast，即是否使用混合精度训练。
		with autocast(args.no_autocast):
			# 得到输出并计算输出和目标之间的差值作为loss
			output = network(source_img)
			loss = criterion(output, target_img)
			# loss = criterion[0](output, target_img) + criterion[1](output, target_img, source_img) * 0.1
			# TV loss
			# tv_loss = criterion[1](output)
			# loss += tv_loss

			# if args.perloss:
			# 	loss2 = criterion[1](output, target_img)
			# 	loss = loss + 0.04 * loss2

		losses.update(loss.item())

		optimizer.zero_grad()
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

	return losses.avg


def valid(val_loader, network):
	PSNR = AverageMeter()

	torch.cuda.empty_cache()

	network.eval()

	for batch in val_loader:
		source_img = batch['source'].cuda()
		target_img = batch['target'].cuda()
		# 该上下文管理器用于禁用梯度计算
		with torch.no_grad():							# torch.no_grad() may cause warning
			output = network(source_img).clamp_(-1, 1)

		mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
		psnr = 10 * torch.log10(1 / mse_loss).mean()
		PSNR.update(psnr.item(), source_img.size(0))

	return PSNR.avg


if __name__ == '__main__':
	print(os.getcwd())
	setting_filename = os.path.join('configs', args.exp, args.model+'.json')
	if not os.path.exists(setting_filename):
		setting_filename = os.path.join('configs', args.exp, 'default.json')
	with open(setting_filename, 'r') as f:
		setting = json.load(f)
	# checkpoint=torch.load('./saved_models_ntire0/indoor/dehazeformer-s.pth')
	checkpoint = None
	network = eval(args.model.replace('-', '_'))()
	# 使用 nn.DataParallel 将模型在多个 GPU 上进行并行计算
	network = nn.DataParallel(network).cuda()

	if checkpoint is not None:
		network.load_state_dict(checkpoint['state_dict'])

	# criterion = []
	# criterion.append(nn.L1Loss())
	# criterion.append(ContrastLoss_res(ablation=False).cuda())

	criterion = nn.L1Loss()
	#TVloss
	# tv_loss = TVLoss().cuda()  # 假设你在使用CUDA
	# criterion.append(tv_loss)
	# if args.perloss:
	# 		vgg_model = vgg16(pretrained=True).features[:16]
	# 		vgg_model = vgg_model.cuda()
	# 		for param in vgg_model.parameters():
	# 			param.requires_grad = False
	# 		criterion.append(PerLoss(vgg_model).cuda())
	# 创建优化器
	if setting['optimizer'] == 'adam':
		optimizer = torch.optim.Adam(network.parameters(), lr=setting['lr'])
	elif setting['optimizer'] == 'adamw':
		optimizer = torch.optim.AdamW(network.parameters(), lr=setting['lr'])
	else:
		raise Exception("ERROR: unsupported optimizer")
	# 余弦退火策略 调整优化器的学习率
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=setting['epochs'],eta_min=setting['lr'] * 1e-2)
	scaler = GradScaler()

	if checkpoint is not None:
		optimizer.load_state_dict(checkpoint['optimizer'])
		scheduler.load_state_dict(checkpoint['lr_scheduler'])
		scaler.load_state_dict(checkpoint['scaler'])
		best_psnr = checkpoint['best_psnr']
		start_epoch = checkpoint['epoch'] + 1
	else:
		best_psnr = 0
		start_epoch = 0

	dataset_dir = os.path.join(args.data_dir, args.dataset)
	train_dataset = PairLoader(dataset_dir, 'train', 'train', 
								setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])

	train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True)
	val_dataset = PairLoader(dataset_dir, 'test', setting['valid_mode'],
							  setting['patch_size'])
	val_loader = DataLoader(val_dataset,
                            batch_size=setting['batch_size'],
                            num_workers=args.num_workers,
                            pin_memory=True)

	save_dir = os.path.join(args.save_dir, args.exp)
	os.makedirs(save_dir, exist_ok=True)

	# if not os.path.exists(os.path.join(save_dir, args.model+'.pth')):
	if checkpoint is not None:
		print('==> Keep training, current model name: ' + args.model)
		# print(network)

		writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))

		# best_psnr = 0
		for epoch in tqdm(range(start_epoch, setting['epochs'] + 1)):

			loss = train(train_loader, network, criterion, optimizer, scaler)

			writer.add_scalar('train_loss', loss, epoch)

			scheduler.step()

			if epoch % setting['eval_freq'] == 0:
				avg_psnr = valid(val_loader, network)
				
				writer.add_scalar('valid_psnr', avg_psnr, epoch)

				if avg_psnr > best_psnr:
					best_psnr = avg_psnr
					print(avg_psnr)
					# torch.save({'state_dict': network.state_dict()},
                	# 		   os.path.join(save_dir, args.model+'.pth'))
					torch.save({'state_dict': network.state_dict(),
								'optimizer': optimizer.state_dict(),
								'lr_scheduler': scheduler.state_dict(),
								'scaler': scaler.state_dict(),
								'epoch': epoch,
								'best_psnr': best_psnr
								},
							   os.path.join(save_dir, args.model + '.pth'))

				writer.add_scalar('best_psnr', best_psnr, epoch)

	else:
		# print('==> Existing trained model')
		# exit(1)
		print('==> Start training, current model name: ' + args.model)
		# print(network)

		writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.exp, args.model))

		best_psnr = 0
		for epoch in tqdm(range(start_epoch, setting['epochs'] + 1)):

			loss = train(train_loader, network, criterion, optimizer, scaler)

			writer.add_scalar('train_loss', loss, epoch)

			scheduler.step()

			if epoch % setting['eval_freq'] == 0:
				avg_psnr = valid(val_loader, network)

				writer.add_scalar('valid_psnr', avg_psnr, epoch)

				if avg_psnr > best_psnr:
					best_psnr = avg_psnr
					print(avg_psnr)
					# torch.save({'state_dict': network.state_dict()},
					# 		   os.path.join(save_dir, args.model+'.pth'))
					torch.save({'state_dict': network.state_dict(),
								'optimizer': optimizer.state_dict(),
								'lr_scheduler': scheduler.state_dict(),
								'scaler': scaler.state_dict(),
								'epoch': epoch,
								'best_psnr': best_psnr
								},
							   os.path.join(save_dir, args.model + '.pth'))

				writer.add_scalar('best_psnr', best_psnr, epoch)
