import torch
from dataset import dataloader_test
import random
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
from constants import *
import os
import argparse

def test(args):
	model = torch.load(os.path.join(args.name, 'generator.pkl') , map_location=torch.device('cpu'))

	images = next(iter(dataloader_test))
	images = torch.reshape(images, (BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE))

	x_offset = random.randint(0, PATCH_SIZE)
	y_offset = random.randint(0, PATCH_SIZE)

	images[:, :, x_offset:x_offset+PATCH_SIZE, y_offset:y_offset+PATCH_SIZE] = 0

	if args.verbose:
		plt.figure(figsize=(8, 8))
		for i in range(16):
			plt.subplot(4, 4, i+1)
			plt.imshow(T.ToPILImage()(images[i]))
		plt.show()

	noise = torch.randn(images.shape[0], 128)
	with torch.no_grad():
		predicted_patches = model(images, noise)
	images[:, :, x_offset:x_offset+PATCH_SIZE, y_offset:y_offset+PATCH_SIZE] = predicted_patches

	if args.verbose:
		plt.figure(figsize=(8, 8))
		for i in range(16):
			plt.subplot(4, 4, i+1)
			plt.imshow(T.ToPILImage()(images[i]))
		plt.show()





def save_results(result, name):
	# Reach the Drive dir
	os.chdir(os.path.join('..'))
	os.chdir(os.path.join('drive'))
	os.chdir( [s for s in os.listdir() if s.startswith('My')][0] )
	if os.path.exists(name) == False:
		os.mkdir(name)
	os.chdir(name)

	# Save files

	# TODO save loss, ecc..
	# ..
	# ..
	# ..

	# Go back to the Git dir
	os.chdir(os.path.join('..'))
	os.chdir(os.path.join('..'))
	os.chdir(os.path.join('..'))
	os.chdir('project-Machine-Learning')
		  


if __name__ == "__main__":        

	parser = argparse.ArgumentParser()
	parser.add_argument("--name", type=str, default='prova')
	parser.add_argument("--verbose", type=bool, default=False)

	args = parser.parse_args()
	
	test(args)