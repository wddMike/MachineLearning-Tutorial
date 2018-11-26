import numpy as np


def readfile(filename):
    data = np.loadtxt(filename,dtype = np.float,delimiter=",")
    print(data)



readfile('ex1data1.txt')