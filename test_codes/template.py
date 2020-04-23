import os
import numpy as np
import json
from PIL import Image
import sys
import cv2

def template(path='template.jpg'):
	data_path = path
	I = Image.open(data_path)
	I = np.asarray(I)
	red = np.array([[I[i,j,0]*2/255.0-1 for j in range(I.shape[1])]for i in range(I.shape[0])])
	green = np.array([[I[i,j,1]*2/255.0-1 for j in range(I.shape[1])]for i in range(I.shape[0])])
	blue = np.array([[I[i,j,2]*2/255.0-1 for j in range(I.shape[1])]for i in range(I.shape[0])])
	
	return red, green, blue
	
print(template())