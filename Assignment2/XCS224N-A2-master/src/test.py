import numpy as np

#sigmoid function # value from 0 to 1
# x = -6
# s = 1./(1 + np.exp(-x))
# print(s)

# val = np.array([[1,2,3],[100,200,300]])
val = np.array([1,2,3])
na = np.newaxis
newval = val[:,na]
# print(val[:,2]) #only 2nd column
# print(val[1,:]) #only 1st row
print(val.shape)
print(newval.shape)
print(newval)
