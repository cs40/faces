import pandas as pd
from numpy import *
from numpy.linalg import norm

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
import random
from sklearn.neighbors import KNeighborsRegressor


#import training set images of baldwin and carrel
def make_set(acts, set_type):
    desired_set = []
    # rand_img = random.sample(range(0, 120), 90)
    for i in acts:
        if set_type == 'training':
            j = 0
            bound = 70
        elif set_type == 'test':
            j = 70
            bound = 80
        if set_type == 'validation':
            j = 80
            bound = 90

        while (j < bound):
            #possible image forrmats
            filename = i + str(j) + '.jpg'
            filename1 = i + str(j) + '.jpeg'
            filename2 = i + str(j) + '.JPG'
            filename3 = i + str(j) + '.png'
            try:
                x_i = imread(set_type + "/" + filename)
            except:
                try:
                    x_i = imread(set_type + "/" + filename1)
                except:
                    try:
                        x_i = imread(set_type + "/" + filename2)
                    except:
                        x_i = imread(set_type + "/" + filename3)
            tup = (x_i, i)
            desired_set.append(tup)
            j +=1
    random.shuffle(desired_set)
    return desired_set

#import training set images of baldwin and carrel
def make_set3(acts, set_type, amt = None):
    desired_set = []
    # rand_img = random.sample(range(0, 120), 90)
    k = 0
    for i in acts:
        if set_type == 'training':
            j = 0

            bound = 68

        elif set_type == 'test':
            if i == 'gilpin':
                j = 68
                bound = 78
            else:
                j = 70
                bound = 80

        if set_type == 'validation':
            if i == 'gilpin':
                j = 78
                bound = 88
            else:
                j = 80
                bound = 90

        while (j < bound):
            #possible image forrmats
            filename = i + str(j) + '.jpg'
            filename1 = i + str(j) + '.jpeg'
            filename2 = i + str(j) + '.JPG'
            filename3 = i + str(j) + '.png'
            try:
                x_i = imread(set_type + "/" + filename)
            except:
                try:
                    x_i = imread(set_type + "/" + filename1)
                except:
                    try:
                        x_i = imread(set_type + "/" + filename2)
                    except:
                        x_i = imread(set_type + "/" + filename3)
            vec = [0]*6
            vec[k] = 1
            np.array(vec)
            tup = (x_i, vec)
            desired_set.append(tup)
            j +=1
        k+=1
    random.shuffle(desired_set)
    return desired_set

def make_set2(acts, set_type, amt = None):
    desired_set = []
    # rand_img = random.sample(range(0, 120), 90)
    for i in acts:
        if set_type == 'training':
            j = 0
            if amt == None:
                if i != 'gilpin':
                    bound = 70
                else:
                    bound = 68
            else:
                bound = amt
        elif set_type == 'test':
            if i != 'gilpin':
                j = 70
                bound = 80
            else:
                j = 68
                bound = 78
        if set_type == 'validation':
            if i != 'gilpin':
                j = 80
                bound = 90
            else:
                j = 78
                bound = 88

        while (j < bound):
            #possible image forrmats
            filename = i + str(j) + '.jpg'
            filename1 = i + str(j) + '.jpeg'
            filename2 = i + str(j) + '.JPG'
            filename3 = i + str(j) + '.png'
            try:
                x_i = imread(set_type + "/" + filename)
            except:
                try:
                    x_i = imread(set_type + "/" + filename1)
                except:
                    try:
                        x_i = imread(set_type + "/" + filename2)
                    except:
                        x_i = imread(set_type + "/" + filename3)

            tup = (x_i, i)
            desired_set.append(tup)
            j +=1
    random.shuffle(desired_set)
    return desired_set




def f(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return sum( (y - dot(theta.T,x)) ** 2)/(2*float(len(y)))

def df(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return -2*sum((y-dot(theta.T, x))*x, 1)/float(len(y))

def grad_descent(f, df, x, y, init_t, alpha, max_iter):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t).reshape(1025, 1)
        # if iter % 500 == 0:
            # print "Iter", iter
            # print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0], t[1], t[2], f(x, y, t))
            # print "Gradient: ", df(x, y, t), "\n"
        iter += 1
    return t



# create x, y and theta
def make_grad_params( input_set, acts ):
    images = np.empty([0, 1024])
    y = []
    for i in input_set:
        images = vstack(    (images, reshape(np.ndarray.flatten(i[0]), [1, 1024])))
        if i[1] == acts[0]:
            y.append(1)
        if i[1] == acts[1]:
            y.append(-1)
    #make sure values are between 0 and 1

    images = images/255.0
    y = np.array(y)
    theta_0 = np.array([0.0] * 1025)
    theta = reshape(theta_0, [1025,1])
    return images, y, theta

def make_multi_params( input_set):
    images = np.empty([0, 1024])
    y = np.empty([0,6])
    for i in input_set:
        images = vstack((images, reshape(np.ndarray.flatten(i[0]), [1, 1024])))
        y = vstack((y, i[1]))
    #make sure values are between 0 and 1

    images = images/255.0
    y = np.array(y)
    theta = np.zeros((1025, 6))

    return images, y, theta


def make_grad_params2( input_set ):
    images = np.empty([0, 1024])
    y = []
    for i in input_set:
        images = vstack((images, reshape(np.ndarray.flatten(i[0]), [1, 1024])))
        if i[1] in ['baldwin', 'carell', 'hader', 'radcliffe', 'butler', 'vartan']:
            y.append(1)
        if i[1] in ['gilpin', 'bracco', 'harmon', 'chenoweth', 'drescher', 'ferrera']:
            y.append(-1)
    #make sure values are between 0 and 1

    images = images/255.0
    y = np.array(y)
    theta_0 = np.array([0.0] * 1025)
    theta = reshape(theta_0, [1025,1])
    return images, y, theta

def validate_results(theta, x, correct_y):
    correct_y = correct_y.tolist()
    result = dot(theta.T, x)
    predicted = []

    for i in range(len(correct_y)):
        if result[:,i] < 0:
            predicted.append(-1)
        else:
            predicted.append(1)
    correct = 0
    for j in range(len(correct_y)):
        if correct_y[j] == predicted[j]:
            correct += 1
    return correct

def validate_mult(theta, x, y):
    predicted = np.empty([0,6])
    x = vstack( (ones((1, x.T.shape[1])), x.T))
    b = dot(theta.T,x)
    for i in b.T:
         max = -10000000
         max_i = 0
         for j in range(len(i)):
             if i[j] > max:
                max = i[j]
                max_i = j
         a = [0 ] * 6
         a[max_i] = 1
         np.array(a)
         predicted = vstack((predicted, a))
    correct = 0
    for i in range(len(predicted)):
        if np.array_equal(predicted[i], y[i]):
            correct +=1
    return correct

def multi_f(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return sum(np.square((y - dot(theta.T,x).T)))

def multi_df(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return 2*dot(x, np.transpose(dot(theta.T,x)-y))

def grad_descent_mulit( multi_df, x, y, init_t, alpha, max_iter):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*multi_df(x, y, t)
        iter += 1
    return t

def part3_4():
    #part 3
    training_set = make_set(['baldwin', 'carell'], 'training')
    training_params = make_grad_params(training_set, ['baldwin', 'carell'])
    training_x = training_params[0].T
    training_y = training_params[1]
    training_t = training_params[2]
    theta = grad_descent(f, df, training_x, training_y , training_t , 0.0010, 3000000)
    theta1 = grad_descent(f, df, training_x, training_y , training_t , 0.00000010,300)
    theta2 = grad_descent(f, df, training_x, training_y , training_t , 0.0010, 3000000)

    training_cost = f(training_x, training_y, theta)
    print "training cost is " + str(training_cost)

    training_x = vstack( (ones((1, training_x.shape[1])), training_x))
    training = (validate_results(theta, training_x, training_y))
    print "training result is " + str(training) + " correct out of 140 images"

    validation_set = make_set(['baldwin', 'carell'], 'validation')
    validation_params = make_grad_params(validation_set, ['baldwin', 'carell'])


    validation_cost = f(validation_params[0].T, validation_params[1], theta)
    print "validation cost is " + str(validation_cost)
    validation_x = vstack( (ones((1, validation_params[0].T.shape[1])), validation_params[0].T))
    validation =  validate_results(theta, validation_x, validation_params[1])
    print "validation result is " + str(validation) + " correct out of 20 images"

    #for part 4
    rand_indices_bald = random.sample(range(0, 70), 2)
    rand_indices_steve = random.sample(range(70, 140), 2)
    two_img_set = []
    two_img_set.append(training_set[rand_indices_bald[0]])
    two_img_set.append(training_set[rand_indices_bald[1]])
    two_img_set.append(training_set[rand_indices_steve[0]])
    two_img_set.append(training_set[rand_indices_steve[1]])
    training_2_params = make_grad_params(two_img_set, ['baldwin', 'carell'])
    two_x = training_2_params[0].T
    theta3 = grad_descent(f, df, two_x, training_2_params[1], training_2_params[2] , 0.0000010, 200)

    # part 4
    theta1 = theta1[1:]
    plt.imshow(reshape(theta1, [32,32]), cmap=cm.coolwarm)
    show()


    theta2 = theta2[1:]
    plt.imshow(reshape(theta2, [32,32]), cmap=cm.coolwarm)
    show()


    theta3 = theta3[1:]
    plt.imshow(reshape(theta3, [32,32]), cmap=cm.coolwarm)
    show()


def part_5():
    #part 5
    #full 70 images per actor/actress
    acts2 = ['bracco', 'gilpin', 'harmon', 'baldwin', 'hader', 'carell']

    training_set2 = make_set2(acts2, 'training')
    l = len(training_set2)
    training_params2 = make_grad_params2(training_set2)
    training_x1 = training_params2[0].T
    training_y1 = training_params2[1]
    training_t1 = training_params2[2]
    theta_full = grad_descent(f, df, training_x1, training_y1, training_t1, 0.0010, 30000)

    # 60 images per actor/actress
    training_set2_60 = make_set2(acts2, 'training', 60)
    l_60 = len(training_set2_60)
    training_params2_60 = make_grad_params2(training_set2_60)
    training_x1_60 = training_params2_60[0].T
    training_y1_60 = training_params2_60[1]
    training_t1_60 = training_params2_60[2]
    theta_60 = grad_descent(f, df, training_x1_60, training_y1_60, training_t1_60, 0.0010, 30000)

    #50 images per actor/actress
    training_set2_50 = make_set2(acts2, 'training', 50)
    l_50 = len(training_set2_50)
    training_params2_50 = make_grad_params2(training_set2_50)
    training_x1_50 = training_params2_50[0].T
    training_y1_50 = training_params2_50[1]
    training_t1_50 = training_params2_50[2]
    theta_50 = grad_descent(f, df, training_x1_50, training_y1_50, training_t1_50, 0.0010, 30000)

     #40 images per actor/actress
    training_set2_40 = make_set2(acts2, 'training', 40)
    l_40 = len(training_set2_40)
    training_params2_40 = make_grad_params2(training_set2_40)
    training_x1_40 = training_params2_40[0].T
    training_y1_40 = training_params2_40[1]
    training_t1_40 = training_params2_40[2]
    theta_40 = grad_descent(f, df, training_x1_40, training_y1_40, training_t1_40, 0.0010, 30000)

    #30 images per actor/actress
    training_set2_30 = make_set2(acts2, 'training', 30)
    l_30 = len(training_set2_30)
    training_params2_30 = make_grad_params2(training_set2_30)
    training_x1_30 = training_params2_30[0].T
    training_y1_30 = training_params2_30[1]
    training_t1_30 = training_params2_30[2]
    theta_30 = grad_descent(f, df, training_x1_30, training_y1_30, training_t1_30, 0.0010, 30000)

    #20 images per actor/actress
    training_set2_20 = make_set2(acts2, 'training', 20)
    l_20 = len(training_set2_20)
    training_params2_20 = make_grad_params2(training_set2_20)
    training_x1_20 = training_params2_20[0].T
    training_y1_20 = training_params2_20[1]
    training_t1_20 = training_params2_20[2]
    theta_20 = grad_descent(f, df, training_x1_20, training_y1_20, training_t1_20, 0.0010, 30000)

    #10 images per actor/actress
    training_set2_10 = make_set2(acts2, 'training', 10)
    l_10 = len(training_set2_10)
    training_params2_10 = make_grad_params2(training_set2_10)
    training_x1_10 = training_params2_10[0].T
    training_y1_10 = training_params2_10[1]
    training_t1_10 = training_params2_10[2]
    theta_10 = grad_descent(f, df, training_x1_10, training_y1_10, training_t1_10, 0.0010, 30000)

    training = []
    val = []

    validation_set2 = make_set2(acts2, 'validation')
    validation_params2 = make_grad_params2(validation_set2)
    validation_x2 = vstack( (ones((1, validation_params2[0].T.shape[1])), validation_params2[0].T))
    validation2 =  validate_results(theta_full, validation_x2, validation_params2[1])

    #test validation and training set on 10 img/per actor training set
    training_x1_10 = vstack( (ones((1, training_x1_10.shape[1])), training_x1_10))
    training2_10 = (validate_results(theta_10, training_x1_10, training_y1_10))
    training.append(training2_10/float(l_10) * 100)
    validation2_10 =  validate_results(theta_10, validation_x2, validation_params2[1])
    val.append(validation2_10/float(60) * 100)

    #test validation and training set on 20 img/per actor training set
    training_x1_20 = vstack( (ones((1, training_x1_20.shape[1])), training_x1_20))
    training2_20 = (validate_results(theta_20, training_x1_20, training_y1_20))
    training.append(training2_20/float(l_20) * 100)
    validation2_20 =  validate_results(theta_20, validation_x2, validation_params2[1])
    val.append(validation2_20/float(60) * 100)

    #test validation and training set on 30 img/per actor training set
    training_x1_30 = vstack( (ones((1, training_x1_30.shape[1])), training_x1_30))
    training2_30 = (validate_results(theta_30, training_x1_30, training_y1_30))
    training.append(training2_30/float(l_30) * 100)
    validation2_30 =  validate_results(theta_30, validation_x2, validation_params2[1])
    val.append(validation2_30/float(60) * 100)

    #test validation and training set on 40 img/per actor training set
    training_x1_40 = vstack( (ones((1, training_x1_40.shape[1])), training_x1_40))
    training2_40 = (validate_results(theta_40, training_x1_40, training_y1_40))
    training.append(training2_40/float(l_40) * 100)
    validation2_40 =  validate_results(theta_40, validation_x2, validation_params2[1])
    val.append(validation2_40/float(60) * 100)

    #test validation and training set on 50 img/per actor training set
    training_x1_50 = vstack( (ones((1, training_x1_50.shape[1])), training_x1_50))
    training2_50 = (validate_results(theta_50, training_x1_50, training_y1_50))
    training.append(training2_50/float(l_50) * 100)
    validation2_50 =  validate_results(theta_50, validation_x2, validation_params2[1])
    val.append(validation2_50/float(60) * 100)

     #test validation and training set on 50 img/per actor training set
    training_x1_60 = vstack( (ones((1, training_x1_60.shape[1])), training_x1_60))
    training2_60 = (validate_results(theta_60, training_x1_60, training_y1_60))
    training.append(training2_60/float(l_60) * 100)
    validation2_60 =  validate_results(theta_60, validation_x2, validation_params2[1])
    val.append(validation2_60/float(60) * 100)

    #test validation and training set on 70 img/per actor training set
    training_x1 = vstack( (ones((1, training_x1.shape[1])), training_x1))
    training2 = (validate_results(theta_full, training_x1, training_y1))
    training.append(training2/float(l) * 100)
    val.append(validation2/float(60 )* 100)

    #part 5 stuff
    acts3 = ['radcliffe', 'butler', 'vartan', 'ferrera', 'chenoweth', 'drescher']
    validation_set_other = make_set2(acts3, 'validation')
    validation_other = make_grad_params2(validation_set_other)
    validation_o_x = vstack( (ones((1, validation_other[0].T.shape[1])), validation_other[0].T))
    validation_o =  validate_results(theta_full, validation_o_x, validation_other[1])
    print validation_o

    t = np.array([10,20,30,40,50,60,70])

    plot(t, val, label='validation')
    plot(t, training, label='training')

    xlabel('Training set size')
    ylabel('Performance in %')
    title('Training Size vs Performance')
    grid(True)
    plt.show()






def part_7():
    acts2 = ['bracco', 'gilpin', 'harmon', 'baldwin', 'hader', 'carell']
    set_p7 = make_set3(acts2, 'training')
    param_7 = make_multi_params(set_p7)
    # print param_7[1].shape#y
    # print param_7[0].shape#x
    # print param_7[2].shape#theta
    multi_theta = grad_descent_mulit( multi_df, param_7[0].T, param_7[1].T, param_7[2], 0.0000010, 30000)
    training = validate_mult(multi_theta, param_7[0], param_7[1])
    print "correct from training is " + str(training)
    print "validate"
    validation_set2 = make_set3(acts2, 'validation')
    validation_params2 = make_multi_params(validation_set2)
    correct = validate_mult(multi_theta, validation_params2[0],validation_params2[1])
    print "The correct number identified from validation is " + str(correct)


    theta1 = multi_theta.T[0][1:]
    plt.imshow(reshape(theta1, [32,32]), cmap=cm.coolwarm)
    show()

    theta2 = multi_theta.T[1][1:]
    plt.imshow(reshape(theta2, [32,32]), cmap=cm.coolwarm)
    show()

    theta3 = multi_theta.T[2][1:]
    plt.imshow(reshape(theta3, [32,32]), cmap=cm.coolwarm)
    show()

    theta4 = multi_theta.T[3][1:]
    plt.imshow(reshape(theta4, [32,32]), cmap=cm.coolwarm)
    show()

    theta5 = multi_theta.T[4][1:]
    plt.imshow(reshape(theta5, [32,32]), cmap=cm.coolwarm)
    show()

    theta6 = multi_theta.T[5][1:]
    plt.imshow(reshape(theta6, [32,32]), cmap=cm.coolwarm)
    show()










if __name__ == '__main__':
    print "started"
    part3_4()
    part_5()
    part_7()




