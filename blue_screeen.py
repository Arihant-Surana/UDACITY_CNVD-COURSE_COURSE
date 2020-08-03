#!/usr/bin/env python
# coding: utf-8

# # Blue Screen

# In[1]:


import matplotlib.pyplot as plt 
import numpy as np
import cv2
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


image = cv2.imread('F:\ObjectDetection\pizza_bluescreen.jpg')
print('This image is :', type(image),
     'with dimension' , image.shape)


# In[4]:


image_copy = np.copy(image)
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
plt.imshow(image_copy)


# # Define the threshold

# In[6]:


lower_blue= np.array([0,0,200])
upper_blue=np.array([250,250,255])


# ### Create a mask

# In[18]:


mask = cv2.inRange(image_copy, lower_blue,upper_blue)
plt.imshow(mask ,cmap='gray')


# In[19]:


# Mask the image to let the pizza show through
masked_image = np.copy(image_copy)

masked_image[mask != 0] = [0, 0, 0]

# Display it!
plt.imshow(masked_image)


# ### Mask and background image

# In[21]:


background_image = cv2.imread('F:\ObjectDetection\space_background.jpg')
background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
crop_background = background_image[0:514,0:816]
crop_background[mask == 0] = [0,0,0]
plt.imshow(crop_background)


# In[22]:


complete_image = masked_image + crop_background
plt.imshow(comple)


# In[ ]:




