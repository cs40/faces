from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc.pilutil import imread,imsave
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
import shutil

actors = ['baldwin', 'bracco', 'butler', 'carell', 'chenoweth', 'drescher', 'ferrera', 'gilpin', 'hader', 'harmon', 'radcliffe', 'vartan']

for i in actors:
    j = 0
    while (j <70):
        filename = i + str(j) + '.jpg'
        filename1 = i + str(j) + '.jpeg'
        filename2 = i + str(j) + '.JPG'
        filename3 = i + str(j) + '.png'
        end = 'jpg'

        if os.path.isfile("cropped/"+filename):

            srcpath = "cropped/" + filename
            dstpath = os.path.join("training/", filename)


        elif os.path.isfile("cropped/"+ filename1):
            end = 'jpeg'
            srcpath = "cropped/" + filename1
            dstpath = os.path.join("training/", filename1)

        elif os.path.isfile("cropped/"+ filename2):
            end = 'JPG'
            srcpath = "cropped/" + filename2
            dstpath = os.path.join("training/", filename2)
        else:
            end = 'JPG'
            srcpath = "cropped/" + filename3
            dstpath = os.path.join("training/", filename3)






        shutil.copyfile(srcpath, dstpath)
        if i == 'gilpin' and j == 68 :
            break
        j+=1

    while (j < 80):
        filename = i + str(j) + '.jpg'
        filename1 = i + str(j) + '.jpeg'
        filename2 = i + str(j) + '.JPG'
        filename3 = i + str(j) + '.png'
        end = 'jpg'

        if os.path.isfile("cropped/"+filename):

            srcpath = "cropped/" + filename
            dstpath = os.path.join("test/", filename)


        elif os.path.isfile("cropped/"+ filename1):
            end = 'jpeg'
            srcpath = "cropped/" + filename1
            dstpath = os.path.join("test/", filename1)

        elif os.path.isfile("cropped/"+ filename2):
            end = 'JPG'
            srcpath = "cropped/" + filename2
            dstpath = os.path.join("test/", filename2)
        else:
            end = 'JPG'
            srcpath = "cropped/" + filename3
            dstpath = os.path.join("test/", filename3)
        shutil.copyfile(srcpath, dstpath)
        if i == 'gilpin' and j == 78:
            break
        j+=1


    while (j < 90):
        filename = i + str(j) + '.jpg'
        filename1 = i + str(j) + '.jpeg'
        filename2 = i + str(j) + '.JPG'
        filename3 = i + str(j) + '.png'
        end = 'jpg'

        if os.path.isfile("cropped/"+filename):

            srcpath = "cropped/" + filename
            dstpath = os.path.join("validation/", filename)


        elif os.path.isfile("cropped/"+ filename1):
            end = 'jpeg'
            srcpath = "cropped/" + filename1
            dstpath = os.path.join("validation/", filename1)

        elif os.path.isfile("cropped/"+ filename2):
            end = 'JPG'
            srcpath = "cropped/" + filename2
            dstpath = os.path.join("validation/", filename2)
        else:
            end = 'JPG'
            srcpath = "cropped/" + filename3
            dstpath = os.path.join("validation/", filename3)
        shutil.copyfile(srcpath, dstpath)
        if (i == 'gilpin' and j == 87):
            break
        j+=1
