from dataset import dataloader_train
from dataset import dataloader_val
from generator import Generator
from discriminator import Discriminator
from constants import *
from tqdm import tqdm
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import torchvision.transforms as T
import torchvision.utils as vutils
import time
import os
import argparse

def train(args):
	EPOCHS = args.epochs	


	lambda_rec = 10  # Peso per la reconstruction loss
	lambda_adv = 1  # Peso per la adversarial loss

	if args.restart_from == 0:
		generator = Generator().to(device)
		discriminator = Discriminator().to(device)
		print(f'Started training using device: {device} - with {EPOCHS} epochs\n')
	else:
		generator, discriminator = load_previous_model(args.name)
		print(f'Restarted training using device: {device} - from {args.restart_from} to {EPOCHS} epochs\n')

	d_opt = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2))
	g_opt = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2))

	loss_fn = nn.BCELoss()

	reconstruction_loss_fn = nn.MSELoss()

	fixed_noise = torch.randn(BATCH_SIZE, Z_DIM, device=device)
	fixed_images = next(iter(dataloader_train))[:BATCH_SIZE].to(device)

	fixed_x_offset = random.randint(0, PATCH_SIZE)
	fixed_y_offset = random.randint(0, PATCH_SIZE)

	g_print=[]
	r_print=[]
	f_print=[]
	vg_print=[]
	vr_print=[]
	vf_print=[]
	epo=[]

	start = time.time()
	for epoch in range(args.restart_from, EPOCHS):
		g_losses = []
		d_losses = []
		real_losses = []
		fake_losses = []
		generator.train()
		discriminator.train()
	
		for image_batch in tqdm(dataloader_train):
			image_batch = image_batch.to(device)
			b_size = image_batch.shape[0]
			discriminator.zero_grad()

			y_hat_real = discriminator(image_batch).view(-1)
			y_real = torch.ones_like(y_hat_real, device=device)
			real_loss = loss_fn(y_hat_real, y_real)
			real_loss.backward()

			# Make part of the image black
			x_offset = random.randint(0, PATCH_SIZE)
			y_offset = random.randint(0, PATCH_SIZE)
			image_batch[:, :, x_offset:x_offset+PATCH_SIZE, y_offset:y_offset+PATCH_SIZE] = 0

			# Predict using generator
			noise = torch.randn(b_size, Z_DIM, device=device)
			predicted_patch = generator(image_batch, noise)

			# Replace black patch with generator output
			image_batch[:, :, x_offset:x_offset+PATCH_SIZE, y_offset:y_offset+PATCH_SIZE] = predicted_patch

			# Predict fake images using discriminator
			y_hat_fake = discriminator(image_batch.detach()).view(-1)

			# Train discriminator
			y_fake = torch.zeros_like(y_hat_fake)
			fake_loss = loss_fn(y_hat_fake, y_fake)
			fake_loss.backward()
			d_opt.step()

			# Compute total discriminator loss and append to list
			d_loss = real_loss + fake_loss
			d_losses.append(d_loss.item())
			real_losses.append(real_loss.item())
			fake_losses.append(fake_loss.item())

			# Train generator
			generator.zero_grad()
			y_hat_fake = discriminator(image_batch).view(-1)
			g_adv_loss = loss_fn(y_hat_fake, torch.ones_like(y_hat_fake))
			reconstruction_loss = reconstruction_loss_fn(predicted_patch, image_batch[:, :, x_offset:x_offset+PATCH_SIZE, y_offset:y_offset+PATCH_SIZE])
			g_loss = lambda_adv * g_adv_loss + lambda_rec * reconstruction_loss
			g_loss.backward()
			g_opt.step()

			# Append generator loss to list
			g_losses.append(g_loss.item())			

		# Validation Loop
		generator.eval()
		discriminator.eval()
		val_g_losses = []
		val_d_losses = []
		val_real_losses = []
		val_fake_losses = []
		
		with torch.no_grad():
			for val_image_batch in tqdm(dataloader_val):
				val_image_batch = val_image_batch.to(device)
				b_size = val_image_batch.shape[0]

				# Validation Discriminator on Real Images
				y_hat_real = discriminator(val_image_batch).view(-1)
				y_real = torch.ones_like(y_hat_real, device=device)
				val_real_loss = loss_fn(y_hat_real, y_real)

				# Make part of the image black in validation set
				x_offset = random.randint(0, PATCH_SIZE)
				y_offset = random.randint(0, PATCH_SIZE)
				val_image_batch[:, :, x_offset:x_offset+PATCH_SIZE, y_offset:y_offset+PATCH_SIZE] = 0

				# Predict using generator
				noise = torch.randn(b_size, Z_DIM, device=device)
				predicted_patch = generator(val_image_batch, noise)

				# Replace black patch with generator output
				val_image_batch[:, :, x_offset:x_offset+PATCH_SIZE, y_offset:y_offset+PATCH_SIZE] = predicted_patch

				# Validation Discriminator on Fake Images
				y_hat_fake = discriminator(val_image_batch).view(-1)
				y_fake = torch.zeros_like(y_hat_fake)
				val_fake_loss = loss_fn(y_hat_fake, y_fake)

				# Validation Generator Loss
				val_reconstruction_loss = reconstruction_loss_fn(predicted_patch, val_image_batch[:, :, x_offset:x_offset+PATCH_SIZE, y_offset:y_offset+PATCH_SIZE])
				val_adv_loss = loss_fn(y_hat_fake, torch.ones_like(y_hat_fake))
				val_g_loss = lambda_adv * val_adv_loss + lambda_rec * val_reconstruction_loss

				# Record validation losses
				val_d_loss = val_real_loss + val_fake_loss
				val_d_losses.append(val_d_loss.item())
				val_real_losses.append(val_real_loss.item())
				val_fake_losses.append(val_fake_loss.item())
				val_g_losses.append(val_g_loss.item())

		fixed_images[:, :, fixed_x_offset:fixed_x_offset+PATCH_SIZE, fixed_y_offset:fixed_y_offset+PATCH_SIZE] = 0
		with torch.no_grad():
			predicted_patches = generator(fixed_images, fixed_noise)
		fixed_images[:, :, fixed_x_offset:fixed_x_offset+PATCH_SIZE, fixed_y_offset:fixed_y_offset+PATCH_SIZE] = predicted_patches
		img = T.ToPILImage()(vutils.make_grid(fixed_images.to('cpu'), normalize=True, padding=2, nrow=4))


		print(f'Epoch {epoch+1}/{EPOCHS}, Generator Loss: {sum(g_losses)/len(g_losses):.3f}, Real Loss: {sum(real_losses)/len(real_losses):.3f}, Fake Loss: {sum(fake_losses)/len(fake_losses):.3f}')
		
		g_print.append(sum(g_losses)/len(g_losses))
		r_print.append(sum(real_losses)/len(real_losses))
		f_print.append(sum(fake_losses)/len(fake_losses))
		epo.append(epoch)

		print(f'Validation Generator Loss: {sum(val_g_losses)/len(val_g_losses):.3f}, Validation Real Loss: {sum(val_real_losses)/len(val_real_losses):.3f}, Validation Fake Loss: {sum(val_fake_losses)/len(val_fake_losses):.3f}')

		vg_print.append(sum(val_g_losses)/len(val_g_losses))
		vr_print.append(sum(val_real_losses)/len(val_real_losses))
		vf_print.append(sum(val_fake_losses)/len(val_fake_losses))

		if epoch % 25 == 0 or epoch==EPOCHS-1:
			save_model(epoch, generator, discriminator, img, args.name)

	os.chdir(os.path.join('..'))
	os.chdir(os.path.join('drive'))
	os.chdir( [s for s in os.listdir() if s.startswith('My')][0] )
	os.chdir(args.name)
	os.mkdir('plots')
	os.chdir('plots')

	#Generator Loss visual
	plt.figure(figsize=(10, 6))
	plt.plot(epo, g_print, marker='o', linestyle='-', color='b', label='gloss')
	plt.xlabel('Epoche')
	plt.ylabel('G_Loss')
	plt.title('Generator Loss')
	plt.grid(True)
	plt.legend()
	plt.savefig('GeneratorLoss')

	#Real Loss visual
	plt.figure(figsize=(10, 6))
	plt.plot(epo, r_print, marker='o', linestyle='-', color='b', label='rloss')
	plt.xlabel('Epoche')
	plt.ylabel('R_Loss')
	plt.title('Real Loss')
	plt.grid(True)
	plt.legend()
	plt.savefig('RealLoss')


	#Fake Loss visual
	plt.figure(figsize=(10, 6))
	plt.plot(epo, f_print, marker='o', linestyle='-', color='b', label='floss')
	plt.xlabel('Epoche')
	plt.ylabel('F_Loss')
	plt.title('Fake Loss')
	plt.grid(True)
	plt.legend()
	plt.savefig('FakeLoss')

	#Validation Generator Loss visual
	plt.figure(figsize=(10, 6))
	plt.plot(epo, vg_print, marker='o', linestyle='-', color='b', label='vgloss')
	plt.xlabel('Epoche')
	plt.ylabel('VG_Loss')
	plt.title('Validation Generator Loss')
	plt.grid(True)
	plt.legend()
	plt.savefig('ValGeneratorLoss')

	#Validation Real Loss visual
	plt.figure(figsize=(10, 6))
	plt.plot(epo, vr_print, marker='o', linestyle='-', color='b', label='vrloss')
	plt.xlabel('Epoche')
	plt.ylabel('VR_Loss')
	plt.title('Validation Real Loss')
	plt.grid(True)
	plt.legend()
	plt.savefig('ValRealLoss')

	#Validation Fake Loss visual
	plt.figure(figsize=(10, 6))
	plt.plot(epo, vf_print, marker='o', linestyle='-', color='b', label='vfloss')
	plt.xlabel('Epoche')
	plt.ylabel('VF_Loss')
	plt.title('Validation Fake Loss')
	plt.grid(True)
	plt.legend()
	plt.savefig('ValFakeLoss')
			
	os.chdir(os.path.join('..'))
	os.chdir(os.path.join('..'))
	os.chdir(os.path.join('..'))
	os.chdir(os.path.join('..'))
	os.chdir('project-Machine-Learning')

	train_time = time.time() - start
	print(f'Total training time: {train_time // 60} minutes')



def save_model(epoch, generator, discriminator, img, name):
	# Reach the Drive dir
	os.chdir(os.path.join('..'))
	os.chdir(os.path.join('drive'))
	os.chdir( [s for s in os.listdir() if s.startswith('My')][0] )
	if os.path.exists(name) == False:
		os.mkdir(name)
	os.chdir(name)
	if os.path.exists('progress') == False:
		os.mkdir('progress')

	# Save files
	torch.save(generator, 'generator.pkl')
	torch.save(discriminator, 'discriminator.pkl')
	img.save(os.path.join('progress', f'epoch_{epoch}.jpg'))
	# TODO save loss, ecc..
	# ..
	# ..
	# ..

	# Go back to the Git dir
	os.chdir(os.path.join('..'))
	os.chdir(os.path.join('..'))
	os.chdir(os.path.join('..'))
	os.chdir('project-Machine-Learning')


def load_previous_model(name):
	os.chdir(os.path.join('..'))
	os.chdir(os.path.join('drive'))
	os.chdir( [s for s in os.listdir() if s.startswith('My')][0] )	
	if os.path.exists(name) == False:
		exit('WRONG MODEL !!!')
	os.chdir(name)
	generator = torch.load('generator.pkl' , map_location=torch.device(device))
	discriminator = torch.load('discriminator.pkl' , map_location=torch.device(device))

	# Go back to the Git dir
	os.chdir(os.path.join('..'))
	os.chdir(os.path.join('..'))
	os.chdir(os.path.join('..'))
	os.chdir('project-Machine-Learning')
	return generator, discriminator

if __name__ == "__main__":        
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", type=int, default=300)
	parser.add_argument("--name", type=str, default='prova')
	parser.add_argument("--restart_from", type=int, default=0)
	args = parser.parse_args()
	
	train(args)
