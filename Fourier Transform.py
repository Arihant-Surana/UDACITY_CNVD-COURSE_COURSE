#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import cv2

get_ipython().run_line_magic('matplotlib', 'inline')

image_stripes = cv2.imread(r"F:\Udacity - Computer Vision Nanodegree\CVND_Exercises-master\1_2_Convolutional_Filters_Edge_Detection\images\stripes.jpg")
image_solid = cv2.imread(r"F:\Udacity - Computer Vision Nanodegree\CVND_Exercises-master\1_2_Convolutional_Filters_Edge_Detection\images\pink_solid.jpg")
image_solid = cv2.cvtColor(image_solid, cv2.COLOR_BGR2RGB)

f, (ax1,ax2) = plt.subplots(1, 2, figsize=(10,5))
ax1.imshow(image_stripes)
ax2.imshow(image_solid)


# In[12]:


gray_stripes = cv2.cvtColor(image_stripes, cv2.COLOR_RGB2GRAY)
gray_solid = cv2.cvtColor(image_solid, cv2.COLOR_RGB2GRAY)

norm_stripes= gray_stripes/255.0
norm_solid = gray_solid/255.0

def ft_image(norm_image):
    f=np.fft.fft2(norm_image)
    fshift = np.fft.fftshift(f)
    frequency_tx = 20*np.log(np.abs(fshift))
    
    return frequency_tx


# In[14]:


f_stripes = ft_image(norm_stripes)
f_solid =ft_image(norm_solid)

f,(ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(20,10))

ax1.set_title("original image")
ax1.imshow(image_stripes)

ax2.set_title("frequency transform image")
ax2.imshow(f_stripes, cmap='gray')

ax3.set_title(" Original image")
ax3.imshow(image_solid)

ax4.set_title("Frequency transfrom image")
ax4.imshow(f_solid, cmap='gray')


# In[15]:



# Read in an image
image = cv2.imread(r'F:\Udacity - Computer Vision Nanodegree\CVND_Exercises-master\1_2_Convolutional_Filters_Edge_Detection\images\birds.jpg')
# Change color to RGB (from BGR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# normalize the image
norm_image = gray/255.0

f_image = ft_image(norm_image)

# Display the images
f, (ax1,ax2) = plt.subplots(1, 2, figsize=(20,10))

ax1.imshow(image)
ax2.imshow(f_image, cmap='gray')


# In[ ]:




