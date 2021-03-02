import pickle
import os
import pandas as pd
from src.parser import *
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from pprint import pprint

def load_dataset(dataset):
	folder = os.path.join(output_folder, dataset)
	if not os.path.exists(folder):
		raise Exception('Processed Data not found.')
	loader = []
	for file in ['train', 'test', 'labels']:
		if dataset == 'SMD': file = 'machine-1-1_' + file
		with open(os.path.join(folder, f'{file}.pkl'), 'rb') as f:
			loader.append(pickle.load(f))
	loader = [i[:, 1:2] for i in loader]
	train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
	test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
	labels = loader[2]
	return train_loader, test_loader, labels

def save_model(model, optimizer, scheduler, epoch, accuracy_list):
	folder = f'checkpoints/{args.model}_{args.dataset}/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/model.ckpt'
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

def load_model(modelname, dims):
	import src.models
	model_class = getattr(src.models, modelname)
	model = model_class(dims).double()
	optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)
	fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
	if os.path.exists(fname):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		checkpoint = torch.load(fname)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1; accuracy_list = []
	return model, optimizer, scheduler, epoch, accuracy_list

def backprop(epoch, model, data, optimizer, scheduler, training = True):
	l = nn.MSELoss(reduction = 'mean' if training else 'none')
	inp, y_true = data[:-1], data[1:]
	if 'VAE' in model.name:
		y_pred, mu, logvar = model(inp)
		MSE = l(y_pred, y_true)
		KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
		loss = MSE + model.beta * KLD
		if training:
			print(f'Epoch {epoch},\tMSE = {MSE},\tKLD = {model.beta * KLD}')
	else:
		y_pred = model(inp)
		MSE = l(y_pred, y_true)
		loss = MSE
		if training:
			print(f'Epoch {epoch},\tMSE = {MSE}')
	if not training:
		return MSE.detach().numpy(), y_pred.detach().numpy()
	else:
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		scheduler.step()
	return loss.item(), optimizer.param_groups[0]['lr']

if __name__ == '__main__':
	train_loader, test_loader, labels = load_dataset(args.dataset)
	model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])

	### Training phase
	print(f'Training {args.model} on {args.dataset}')
	trainD, testD = next(iter(train_loader)), next(iter(test_loader))
	num_epochs = 10
	for e in range(epoch+1, epoch+num_epochs+1):
		lossT, lr = backprop(e, model, trainD, optimizer, scheduler)
		accuracy_list.append((lossT, lr))
	save_model(model, optimizer, scheduler, e, accuracy_list)
	torch.zero_grad = True
	model.eval()

	### Testing phase
	print(f'Testing {args.model} on {args.dataset}')
	loss, y_pred = backprop(0, model, trainD, optimizer, scheduler, training=False)

	### Plot curves
	plotter(f'{args.model}_{args.dataset}', trainD, y_pred, loss, labels)

	### Scores
	df = pd.DataFrame()
	lossT, _ = backprop(0, model, trainD, optimizer, scheduler, training=False)
	for i in range(loss.shape[1]):
		lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
		result = pot_eval(lt, l, ls[1:])
		df = df.append(result, ignore_index=True)
	print(df)