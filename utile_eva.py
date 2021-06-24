import os, sys
import PIL.Image as Image
import tarfile
import shutil
import numpy as np
import math
import random
import cv2
import time
from numba import jit,njit

def correlation(img,kernal):
    # function used by ssim
    kernal_heigh = kernal.shape[0]
    kernal_width = kernal.shape[1]
    cor_heigh = img.shape[0] - kernal_heigh + 1
    cor_width = img.shape[1] - kernal_width + 1
    result = np.zeros((cor_heigh, cor_width), dtype=np.float64)
    for i in range(cor_heigh):
        for j in range(cor_width):
            result[i][j] = (img[i:i + kernal_heigh, j:j + kernal_width] * kernal).sum()
    return result

def gaussian_2d_kernel(kernel_size=11, sigma=1.5):
    # function used by ssim
    kernel = np.zeros([kernel_size, kernel_size])
    center = kernel_size // 2

    if sigma == 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8

    s = 2 * (sigma ** 2)
    sum_val = 0
    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / s)
            sum_val += kernel[i, j]
    sum_val = 1 / sum_val
    return kernel * sum_val

def ssim(distorted_image,original_image,window_size=11,gaussian_sigma=1.5,K1=0.01,K2=0.03,alfa=1,beta=1,gama=1):
    # Compute ssim of two images( the images must be gray scale 1 channel image)
    distorted_image=np.array(distorted_image,dtype=np.float64)
    original_image=np.array(original_image,dtype=np.float64)
    if not distorted_image.shape == original_image.shape:
        raise ValueError("Input Imagees must has the same size")
    if len(distorted_image.shape) > 2:
        raise ValueError("Please input the images with 1 channel")
    kernal=gaussian_2d_kernel(window_size,gaussian_sigma)

    ux=correlation(distorted_image,kernal)
    uy=correlation(original_image,kernal)
    distorted_image_sqr=distorted_image**2
    original_image_sqr=original_image**2
    dis_mult_ori=distorted_image*original_image
    uxx=correlation(distorted_image_sqr,kernal)
    uyy=correlation(original_image_sqr,kernal)
    uxy=correlation(dis_mult_ori,kernal)
    ux_sqr=ux**2
    uy_sqr=uy**2
    uxuy=ux*uy
    sx_sqr=uxx-ux_sqr
    sy_sqr=uyy-uy_sqr
    sxy=uxy-uxuy
    C1=(K1*255)**2
    C2=(K2*255)**2

    if(alfa==1 and beta==1 and gama==1):
        ssim=(2*uxuy+C1)*(2*sxy+C2)/(ux_sqr+uy_sqr+C1)/(sx_sqr+sy_sqr+C2)
        return np.mean(ssim)

    l=(2*uxuy+C1)/(ux_sqr+uy_sqr+C1)
    l=l**alfa

    sxsy=np.sqrt(sx_sqr)*np.sqrt(sy_sqr)
    c=(2*sxsy+C2)/(sx_sqr+sy_sqr+C2)
    c=c**beta

    C3=0.5*C2
    s=(sxy+C3)/(sxsy+C3)
    s=s**gama
    ssim=l*c*s
    return np.mean(ssim)

def psnr(img1, img2):
    # Compute psnr of two images(The input can be RGB color image)
    img1 = np.array(img1)
    img2 = np.array(img2)
    mse = np.mean( (img1/255. - img2/255.) ** 2 )
    if mse < 1.0e-10:
      return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def mse(img1, img2):
    # Compute mse of two images(The input can be RGB color image)
    img1 = np.array(img1)
    img2 = np.array(img2)
    return np.mean( (img1/255. - img2/255.) ** 2 )

"""
target = None
ref = None

for i, img in enumerate(imagelist):
    if i == 1:
        target = Image.open('/Users/chan/Desktop/evaluate/image/'+str(img))
        target_data = np.array(target)

        #target = random_noise(target, 1000)
    else:
        ref = Image.open('/Users/chan/Desktop/evaluate/image/'+str(img))


print(psnr(target, ref, 224))
print(psnr1(target, ref))
print(psnr2(target, ref))
print(ssim(target, ref))
"""
