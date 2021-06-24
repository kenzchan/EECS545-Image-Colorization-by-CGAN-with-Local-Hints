import os, sys
import PIL.Image as Image
import tarfile
import shutil
import numpy as np
import math
import random

size = 224, 224

inp_path = '/Users/chan/Desktop/evaluate/image' 
imagelist = os.listdir(inp_path)

size = len(imagelist)


for i, img in enumerate(imagelist):

	tmp = img
	idx, state = tmp.split('_')
	if state == 'M' or state == 'I':
		continue
	img = Image.open('/Users/chan/Desktop/evaluate/image/'+str(tmp))

	img_resize_lanczos = img.resize((224, 224), Image.LANCZOS)
	print(img_resize_lanczos)
	img_resize_lanczos.save('/Users/chan/Desktop/evaluate/image_resize/'+str(tmp))


def psnr1(img1, img2):
	img1 = np.array(img1)
	img2 = np.array(img2)

	mse = np.mean((img1 - img2) ** 2 )
	if mse < 1.0e-10:
	  return 100
	return 10 * math.log10(255.0**2/mse)
 
def psnr2(img1, img2):
	img1 = np.array(img1)
	img2 = np.array(img2)
	mse = np.mean( (img1/255. - img2/255.) ** 2 )
	if mse < 1.0e-10:
	  return 100
	PIXEL_MAX = 1
	return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

imagelist = os.listdir("/Users/chan/Desktop/evaluate/image_resize/") 
#for infile in imagelist:
	#print(infile)

size = len(imagelist)

ground_list = ["1_G.png", "2_G.png", "3_G.png", "4_G.png", "5_G.png", "6_G.png"]

s_ave_A1 = 0
s_ave_A2 = 0
s_ave_I = 0

s_ave_H1 = 0
s_ave_H2 = 0
s_ave_H3 = 0

i_ave_A1 = 0
i_ave_A2 = 0
i_ave_I = 0

i_ave_H1 = 0
i_ave_H2 = 0
i_ave_H3 = 0

for img in imagelist:

	tmp = img
	idx, state = tmp.split('_')
	state, _ = state.split('.')
	tmp_g_img = ground_list[int(idx)-1]

	g_img = Image.open('/Users/chan/Desktop/evaluate/image_resize/'+str(tmp_g_img))
	img = Image.open('/Users/chan/Desktop/evaluate/image_resize/'+str(tmp)).convert('RGB')
	psnr = psnr1(img, g_img)
	img = np.mean(img, axis=2)
	g_img = np.mean(g_img, axis=2)

	print("PSNR between" + tmp + "and" + tmp_g_img + ":", psnr)


	if state == 'A1':
		if idx == '1' or idx == '2':
			s_ave_A1 += psnr
		else:
			i_ave_A1 += psnr
	elif state == 'A2':
		if idx == '1' or idx == '2':
			s_ave_A2 += psnr
		else:
			i_ave_A2 += psnr
	elif state == 'H1':
		if idx == '1' or idx == '2':
			s_ave_H1 += psnr
		else:
			i_ave_H1 += psnr	
	elif state == 'H2':
		if idx == '1' or idx == '2':
			s_ave_H2 += psnr
		else:
			i_ave_H2 += psnr	
	elif state == 'H3':
		if idx == '1' or idx == '2':
			s_ave_H3 += psnr
		else:
			i_ave_H3 += psnr	
	else:
		if idx == '1' or idx == '2':
			s_ave_I += psnr
		else:
			i_ave_I += psnr

print("Ave_PSNR of A1 on S:", s_ave_A1/2)
print("Ave_PSNR of A2 on S:", s_ave_A2/2)
print("Ave_PSNR of H1 on S:", s_ave_H1/2)
print("Ave_PSNR of H2 on S:", s_ave_H2/2)
print("Ave_PSNR of H3 on S:", s_ave_H3/2)

print("Ave_PSNR of A1 on I:", i_ave_A1/4)
print("Ave_PSNR of A2 on I:", i_ave_A2/4)
print("Ave_PSNR of H1 on I:", i_ave_H1/4)
print("Ave_PSNR of H2 on I:", i_ave_H2/4)
print("Ave_PSNR of H3 on I:", i_ave_H3/4)



