import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import pi,newaxis, ogrid
from numpy.lib import recfunctions as rfn

#==================================================================================
#                     SOURCE

#This document is based on the source of the documentation of Numpy:
#https://docs.scipy.org/doc/numpy/user/quickstart.html

#Cleared: Basic statistics, Algebra, Operations, ordering, questions, manipulations, conversions, array creation

#This document is meant to go under the designation of being illustrative of a learning process.

#I in no way, shape or form - claim to own the represented material or intend to eskew it to be
#of my own owning.

#All rights reserved to The SciPy community.

#===================================================================================
#====================================================================================
#
#                    INDEXING
#                  Index 1.0 - Basics
#                  Index 1.1 - Splits
#                  Index 1.2 - No Copies
#                  Index 1.3 - Shallow Copy/View
#                  Index 1.4 - Deep Copy
#                  Index 1.5 - Broadcasting
#                  Index 1.6 - Indexing
#                  Index 1.7 - Reductions
#                  Index 1.8 - Structured Arrays
#                  Index 1.9 - Mean/STD/Variance
#                  Index 2.0 - Linear Algebra
#                  Index 2.1 - Allclose Interaction
#                  Index 2.2 - SVD
#                  Index 2.3 - VDot
#                  Index 2.4 - Choose
#                  Index 2.5 - Compression
#                  Index 2.6 - Cumprod/Cumsum/Inner Product
#                  Index 2.7 - Tensordot
#                  Index 2.8 - Numpy.ndarray.fill
#                  Index 2.9 - np.imag
#                  Index 3.0 - np.prod
#                  Index 3.1 - np.put
#                  Index 3.2 - np.putmask                 
#                  Index 3.3 - numpy.real
#                  Index 3.4 - numpy.sum
#                  Index 3.5 - numpy.argmax
#                  Index 3.6 - np.unravel_index
#                  Index 3.7 - numpy.argmin
#                  Index 3.8 - np.argsort
#                  Index 3.9 - numpy.ptp  
#                  Index 4.0 - numpy.searchsorted
#                  Index 4.1 - np.sort           
#                  Index 4.2 - np.all
#                  Index 4.3 - numpy.any
#                  Index 4.4 - numpy.nonzero
#                  Index 4.5 - numpy.where()
#                  Index 4.6 - numpy.array_split()
#                  Index 4.7 - numpy.column_stack
#                  Index 4.8 - numpy.concatenate
#                  Index 4.9 - numpy.diagonal
#                  Index 5.0 - numpy.dsplit
#                  Index 5.1 - numpy.dstack
#                  Index 5.2 - numpy.hsplit
#                  Index 5.3 - numpy.hstack
#                  Index 5.4 - numpy.ndarray.item
#                  Index 5.5 - numpy.newaxis
#                  Index 5.6 - numpy.ravel
#                  Index 5.7 - np.repeat
#                  Index 5.8 - np.reshape
#                  Index 5.9 - np.resize
#                  Index 6.0 - numpy.squeeze
#                  Index 6.1 - numpy.swapaxes
#                  Index 6.2 - Slicing and np.index_exp
#                  Index 6.3 - numpy.take
#                  Index 6.4 - numpy.transpose
#                  Index 6.5 - numpy.vsplit
#                  Index 6.6 - numpy.vstack
#                  Index 6.7 - numpy.astype
#                  Index 6.8 - numpy.atleast_1d
#                  Index 6.9 - numpy.atleast_2d
#                  Index 7.0 - numpy.atleast_3d
#                  Index 7.1 - numpy.mat
#                  Index 7.2 - numpy.arange
#                  Index 7.3 - numpy.array
#                  Index 7.4 - numpy.copy
#                  Index 7.5 - numpy.empty
#                  Index 7.6 - numpy.empty_like
#                  Index 7.7 - numpy.eye
#                  Index 7.8 - numpy.fromfile
#                  Index 7.9 - numpy.fromfunction
#                  Index 8.0 - numpy.identity
#                  Index 8.1 - numpy.linspace
#                  Index 8.2 - numpy.mgrid
#                  Index 8.3 - numpy.ogrid
#                  Index 8.4 - numpy.ones
#                  Index 8.5 - numpy.ones_like
#                  Index 8.6 - numpy.zeros
#                  Index 8.7 - numpy.zeros_like
#
#
#
#====================================================================================
#
#                  POST MORTEM
#       -------------------------
#   DIFFICULTIES: 
#   There were some parts of this specific tutorial and overview that i missed out on.
#   Whilst the actual mathematics and what not were not of very hard difficulty,
#   there were some parts that were lost unto me.
#   
#   Most notably was the higher level of Mathematics, that were kind of hard for me to figure out.
#   All i had to go on, was documentation outside of the actual function backgrounds and documentation.
#  
#   WHAT I LEARNED:
#   I felt like i learned a lot of Matris dynamics, a bit of implicit conversion, broadcasting
#   Some mathematical operations, some pattern recognition
#
#   There also were some instances of where functional subsectioning and lambda partitionings 
#   in the documentation were to be of something relevant to talk about - But, that's more on the level
#   of that i recall it bypassingly.
#
#   WHAT WENT WRONG:
#   I think partially, failure to fully realize some of the higher levels of dynamics along with lack of integration
#   of speed optimization, caused a failure of accounting of deeper dynamics.
#  
#   past this, i think application of real modelling in terms of a real project, needs to be done.
#   And partially, some better balance must be applied in the amount of time done/taken.
#
#   WHAT'S NEXT:
#   Next up, i want to look into of how to perform better structures of SQL queries
#
#=====================================================================================
#       TIME TAKEN: 2-3 Weeks approximately
#=====================================================================================



#BASICS - Index 1.0
print("=================== SHOWCASING basics ======================\n")

a = np.arange(15).reshape(3, 5) #Np produces an evenly spaced range within the given interval
#The arguments are:
#start - number,optional - Start of the interval. Defaults to 0
#stop: end of interval
#step: the distance moving between
#dtype - type of the output array, if not defaulted, infers data type

#returns - ndarray - Array of evenly spaced values
#for floating point, it rounds up - cause overflow errors in floating point precision
#the length of the result is ceil((stop - start)/step)

#In terms of the shape call, it gives the dimensions of the Matris
print("This is the shape of a: " +  str(a.shape) + "\n")

#In the meantime, the ndim is the amount of dimensions
x = np.array([1,2,3]) #Yields 1 axis and 3 elements, contains one row, note the implicit [] containment
b = np.array([1,2]) #Yields 1 axis, 2 elements
c = np.array(([1,2],[3,4])) #2*2, yields size 4, 2 axises, 2 elements each
d = np.array(([1,2],[3,4],[4,5])) #3*2, yields size 6, 3 axises, 2 elements each
print("This is the dimensions of X: "  + str(x.ndim))

y = np.zeros((3,3,3)) #Yields size 27, because 3*3*3, because 3 axises, 3 elements each, zeros just designating elements of 0 to each row
print("This is y, in terms of being printed: \n " + str(y))
#print(str(y))
yx = np.zeros((3,3,3,3)) #Yields size 81, or should, because 4 []'s, so 3*3*3*3
print("This is the dimensions of Y: " + str(y.ndim) + "\n")

#To infer the type, we can write dtype
print("This is x's dtype name: " + str(x.dtype.name))
print("This is y's dtype name: " + str(y.dtype.name) + "\n") #Note that, when we go to higher containments and floating points, we store a larger value in terms of memory allocation

print("This is x's dtype: " + str(x.dtype)) #type accessing yields showcasing of that it's just a typing
print("This is y's dtype: " + str(y.dtype) + "\n") #type accessing yield showcasing of that it's just a typing, as well, in terms of Y, showcasing type

#In terms of one array element in length, we can deduce that from itemsize calls
print("This is one array element in terms of length, for X : "  + str(x.itemsize)) #This reflect backs unto their typing, as it's the byte size reflecting the typing
print("This is one array element in terms of length, for Y : "  + str(y.itemsize) + "\n")

#The size of the matrises, is the product of all their elements, so if the matris is 3,5,2 - it's 3*5*2 (30, because 15*2)
print("This is the size of X, the product of the matris: " + str(x.size))
print("This is the size of B, the product of the matris: "  + str(b.size))
print("This is the size of C, the product of the matris: " + str(c.size) + "\n")
print("This is the size of D, the product of the matris: " + str(d.size))
print("This is the size of Y, the product of the matris: " + str(y.size)) #24, because 2*3*4 The multiplication is implicit of all the elements, contra level of size based on dimension
#So, if there is 3 axises, it's n*n*n, where n is the respective element of the axis

print("This is the size of YX, the product of the matris: " + str(yx.size) + "\n") #yields 81, 4 implicit levels

print("This is the type of X : " + str(type(x))) #As can be showcased, this is the typing of different structures
print("This is the type of Y : " + str(type(y)) + "\n")

#In terms of implicit typing, the conversion of typing is done per designation of elements, as showcased:
testINT = np.array([2,3,4])
print("This is the array, called Test: " +  str(testINT))

testFLOAT = np.array([2.1, 3.2, 4.3])
print("This is the array, called Test2: " + str(testFLOAT) + "\n")

print("This is the type of X : " + str(testINT.dtype)) #Showcasing of int type allocations
print("This is the type of Y : " + str(testFLOAT.dtype) + "\n") #Showcasing of float implicit type conversion

#An important aspect in terms of the parsing of the containers, is designation of level in writing:
willWork = np.array([1,2,3,4]) #Will work, 1 implicit axis, 4 elements
#wontWork = np.array(1,2,3,4) Will not work, because no implicit containment of axis level in terms of []

tryingNew = np.array([(1.5,2,3), (4,5,6)]) #The implicit type casting is to convert all to floating points, due to one element being floating point
print("This is the type of tryingNew: " + str(tryingNew.dtype))
print("Showcasing: " + str(tryingNew))

print("This is the size of the Matris: " + str(tryingNew.size) + "\n") #6, because 2 axises, 3 elements - 2*3

#If we wish, we can denote a specific typing and structure - to adhere to how we wish the Matris to be built
irrational = np.array([[1,2],[3,4]], dtype=complex)
print("This is a irrationally constructed Matris: " + str(irrational) + "\n")

#The above is a operation of upcasting, in terms of type designation.
#Without proper type declaration, defaulting occurs to what is nessecary to hold the typing.
#To downtype, use .astype(<type>) method

#We can also explicitly force a minimum level of dimensions required in terms of Matris structures
showcasing = np.array([1,2,3], ndmin=2)
print(str(showcasing))
print("This is the showcasing, to show 2 dimensions of the explicit typing declaration: " + str(showcasing) + " (it yields 2 [[, thus it's 2 axises) \n")

#Attempting to showcase partial partitioned numerals
x = np.array([(1,2), (3,4)],dtype=[('a', '<i4'), ('b', '<i4')]) #In terms of typing, <i4 is a 32 bit int, <i8 is a 64 bit int
#<f4 is a 32 bit float, <f8 is a 64 bit float


print("This is showcasing of x['a'] accessings: " + str(x['a']))
print("This is showcasing of x['b'] accessings: " + str(x['b']))

print("The total size of x['a'] is : " + str(x['a'].size)) #2, since the subpartitioning is 2 elements, 1 axis, 2*1
print("The total size of x['b'] is : " + str(x['b'].size) + "\n") #2, as well

#The default typing in terms of initializations of using the zeros() function call, is float64
zeros = np.zeros((3,4))
print("Showcasing that default typing initialization is float64: \n " + str(zeros))
print("The type of zeros is: " + str(zeros.dtype) + " (utilizing dtype attribute of zeros) \n")

#We can also utilize calling ones, to initialize with all elements being 1 - and declare typing
ones = np.ones((3,4), dtype=np.int16)
print("Showcasing that typing initialization is specified to int16: \n" + str(ones))
print("The type of ones is: " + str(ones.dtype) + " (utilizing dtype attribute of ones) \n")

#We can also call empty, where we initialize with random states based on state of memory
empty = np.empty((2,3))
print("Showcasing the typing of empty initialization, typing is float64 if not declared: \n" + str(empty))
print("The type of empty is: " + str(empty.dtype) + " (utilizing dtype attribute of empty) \n")

#We can also create a range of numbers, with the input params of <START, END, STEP> (and a few other args)
sequence = np.arange(10,30,5) #step 5, range from 10 to 30
print("Showcasing the ranging of np.arange: \n" + str(sequence) + "\n")

#We can also utilize float arguments for that
floatSeq = np.arange(10, 20, 0.5) #step 0.5, from 10 to 20
print("Showcasing the ranging of np.arange with float points: \n" + str(floatSeq) + "\n")

#An issue with float point argument inputs, is that it renders no guarantee on amount of elements obtained, due to
#floating conversion. However, we can use linspace to circumvent this, with a designated amount to return:

linspace = np.linspace(0, 2, 9) #9 numbers from 0 to 2
print("Showcasing linspace result: \n" + str(linspace) + "\n")

#We can also utilize notations akin to pi and others, to integrate unto the functionings
x = np.linspace(0, 2*pi, 100) #can evaluate a function at a lot of points
print("Showcasing of linspace with a 100 points of intervals in terms of 0 to 2*pi: \n " + str(x) + "\n")

#We can then modify the derived space unto other integrations
f = np.sin(x)
print("Showcasing transformation of x evaluations integrated unto sin transformation: \n " + str(f) + "\n")

#In terms of formatting, single dimensional arrays are printed as rows
#Bidimensionals are matrices
#Tridimensionals are lists of matrices

oneD = np.arange(6) #a 1d array of range in 0-5
print("Showcasing single dimensional array: " + str(oneD))

twoD = np.arange(12).reshape(4,3) #a 2d array of range in 0-11
print("Showcasing two dimensional array: \n " + str(twoD))

threeD = np.arange(24).reshape(2,3,4) #a 3d array of range in 0-23
print("Showcasing three dimensional array: \n " + str(threeD) + "\n")

#If for some reason an array would be too large, the central part is truncated
print("Showcasing of truncation of middle in too large array: \n " + str(np.arange(10000)) + "\n")

print("Showcasing of truncation of multidimensional arrays: \n " + str(np.arange(10000).reshape(100,100)) + "\n")

#If we wish to circumvent this, we can use np.set_printoptions(threshold=np.nan)

#Arithmetic operations on arrays apply elementwise. A fresh new array is created and filled with the result.

#Initialize arrays
a = np.array([20,30,40,50])
b = np.arange(4)

#assign the result of operations
c = a - b #The result is 20 -0, 30-1, 40-2, 50-3
print("Showcasing the result of implicit results of operations from a (np.array([20,30,40,50])) - b (np.arange(4)): \n" + str(c) + "\n")

#Implicit results of operation performances
b = b**2
print("Showcasing inherent result of a range of 0,1,2,3 being raised to the power of two: \n " + str(b) + "\n")

#Implicit results of chains of operations
a = 10*np.sin(a)
print("Showcasing chain of operations results on a: \n " + str(a) + "\n")

#Implicit results of comparison operations
result = a<1
print("Showcasing the piecewise operations done by comparisons in case of 'a<1' : \n" + str(result) + "\n")

#In terms of NumPy, the * operator causes elementwise interaction.
A = np.array([[1,1],[0,1]])
B = np.array([[2,0],[3,4]])

print("Base is: \n A = np.array([[1,1],[0,1]]) \n B = np.array([[2,0],[3,4]]) \n")

C = A * B #Performs the operations of 2*1, 1*0, 3*0, 4*1, giving 2,0,0,4
print("Showcasing the results of * operator interactions from A * B: \n " + str(C) + " \n")

#If we wish to access the matrix product, we use @ or .dot operations
#C = A @ B This only works in Python >= 3.5, this was done in 2.7.9
#print("Showcasing the results of @ operator interactions: \n " + str(C) + " \n")
o = np.array([[0,2.5],[2.5,2.5]]) #Gives 50 if all 5,  
p = np.array([[2.5,2.5],[0,2.5]]) #If all of these are 0, the result is zero
#
#
# o    [ [ a , b ] , [ c, d ] ]
# p    [ [ x , y ] , [ z , o] ]
#--------vvvvvvvvvvvv--------- Dot product of Matrix Multiplication
#[ [ x*a + z*b , y*a  + o*b ]  [ x*c + z*d , y*c + o*d ] ]
 

C = A.dot(B)
D = o.dot(p)
print("Showcasing the results of the Matrix product, using the .dot() function with A.dot(B): \n " + str(C) + " \n")


#We can utilize operators to operate on already in place arrays, instead of creating new ones:
a = np.ones((2,3), dtype=int)
print("Showcasing the assignment of the first Matris without modification: \n" + str(a) + "\n")
b = np.random.random((2,3))

a *= 3
print("Showcasing after the assignment of the first Matris with modification: \n" + str(a) + "\n")

#Showcasing concatenation
b += a
print("Showcasing after concatenation of first Matris plus a Randomly initialized one: \n" + str(b) + "\n")

#However, implicit typecasting is not comitted in terms of concatenation, in terms of a += b

#In terms of operating with chain concatenation of typing, we perform upcasting

#We go from int32
a = np.ones(3, dtype=np.int32)
b = np.linspace(0,pi,3) #Toupcasting to float64
print("When concatenated typing is now converted to: " + str(b.dtype.name) + " B's typing, from having been A's : " + str(a.dtype.name) + "\n")

#Add them together for later operations
c = a+b

#Next we create a complex typing concatenation
d = np.exp(c*1j)
print("Showcasing the new Matris result from exp. concatenation: \n " + str(d) + "\n")
#To showcase the typing result
print("The result typing is: \n " + str(d.dtype.name) + "\n")

#Unary operations akin to Computing the sum of all the elements in the array, are implemented as methods of the ndarray class

#Initialize a random matris
a = np.random.random((2,3))
print("Showcasing the random initialized Matris: \n " + str(a) + "\n")

#The operations are subclassings of the ndarray class

#Perform summing
print("Showcasing the result from sumation: " + str(a.sum()) + "\n")

#Access minimal value
print("Showcasing the smallest value: " + str(a.min()) + "\n")

#Access largest value
print("Showcasing the largest value: " + str(a.max()) + "\n")

#Normally, operations occur as if the entire formatting is a list. We can specify this with
#axis parameter declarations

b = np.arange(12).reshape(3,4)
print("Showcasing of the initialized Matris: \n " + str(b) + "\n")

#To run sum of each column
b2 = b.sum(axis=0)
print("The result of sumation with axis 0, suming each column: " + str(b2) + "\n")

#To run min of each column
b3 = b.min(axis=1)
print("The result of picking the minimum with axis 1, taking it at each column: " + str(b3) + "\n")

#To run the cummulative sum along each row
c = b.cumsum(axis=1)
print("The result of cumulative sum of each row: \n " + str(c) + "\n")

#In terms of mathematical functions akin to sin, cos, exp - they are called universal functions.

#These include: all, any, apply_along_axis, argmax, argmin, argsort, average, bincount, ceil, clip,
#conj, corrcoef, cov, cross, cumprod, cumsum, diff, dot, floor, inner, inv, lexsort, max, maximum
#mean, median, min, minimum, nonzero, outer, prod, re, round, sort, std, sum, trace, transpose
#var,vdot,vectorize, where

#In terms of 1d arrays, they can be indexed, sliced and iterated - like Lists/other Py sequences

a = np.arange(10)**3
print("Showcasing of the basic structure of a: \n" + str(a) + "\n")

#Accessing indexes
a2 = a[2]
print("Showcasing basic accessing of index a[2]: " + str(a2) + "\n")

#Showcasing of slicing
a2 = a[2:5]
print("Showcasing of slicing operations of a[2:5]: \n" + str(a2) + "\n")

#Showcasing of slicing with movement
copyA = a
copyA[:6:2] = -1000

print("Showcasing entire list after modification: " + str(copyA) + "\n")

#Reverse the list
copyA = copyA[::-1]
print("Showcasing of reversal of list: " + str(copyA) + "\n")

#Multidim arrays can have one index per axis, the indices are given in a tuple separated by commas:

def f(x,y):
    return 10*x+y

b = np.fromfunction(f,(5,4),dtype=int) #Iterate forth a Matris that is 5*4, accounting for types being int, based on function
print("Showcasing of b: \n" + str(b) + "\n")

#showcasing of accessing indexes
print("Accessing b[2,3]: " + str(b[2,3]) + "\n")

#Showcasing of slicing
print("Showcasing of b before slice: \n" + str(b) + "\n")
#Access first to 5:th element of the 2:nd column of the 5*4 matris
b2 = b[0:5, 1]
print("Showcasing of b after slice: \n" + str(b2) + "\n")

#Access all elements of the 2:nd column of the 5*4 matris
b2 = b[:,1] 
print("Showcasing of b after slice, looks identical - but accesses all elements, instead of 1-5:th: \n " + str(b2) + "\n")

#Access entire 2:nd and 3:rd row
b2 = b[1:3, : ]
print("Showcasing of b after slicing rows: \n " + str(b2) + "\n")

#Index accessing works as per usual, in terms of last index and what not
b2 = b[-1]
print("Showcasing of accessing last row by -1 index of b: " + str(b2) + "\n")

#In terms of trailing completion, we can utilize ... notations in listings, akin to:
# x[1,2,...] - Which will be equivalent to 1,2,:,:,:

#we begin by setting up our matris that we will work on
c = np.array([[[ 0, 1, 2 ],
                  [ 10, 12, 13]],
                  [[100,101,102],
                   [110,112,113]]])

#Showcase shape

print("The shape of c is: " + str(c.shape))
print("Showcasing of c \n " + str(c) + "\n")

#Access trailing point of from second row
print("Showcasing of trailing access from second row: \n " + str(c[1,...]) + "\n")

#Access trailing point of last element in each row
print("Showcasing of trailing access for last element in each row and column: \n" + str(c[...,2]) + "\n")

#We can also iterate in terms of rows
print("Showcasing of iterating of rows: \n")
for row in c:
    print(str(row))
print("\n")
#We can also iterate over individual elements
print("Showcasing iteration of individual elements of c: \n")
for element in c.flat:
    print(str(element))

print("\n")

#We can also do some shape manipulation in terms of the Matrises
a = np.floor(10*np.random.random((3,4)))
print("Showcasing of a randomly initialized Matris: \n " + str(a) + "\n")
print("Showcasing of size: " + str(a.shape))

#We can run modifications in terms of modified Matrises - but we do not modify the original by the calls
b = a.ravel() #Flattens the array
print("Showcasing of flattening of the random Matris: \n" + str(b) + "\n")

c = b.reshape(6,2) #We can reshape the Matris, if we wish
print("Showcasing of reformation of the same Matris: \n" + str(c) + "\n")

#We can also transpose the matris
d = c.T #Transposition
print("Showcasing of transposition in terms of the Matris: \n" + str(d) + "\n")

print("Showcasing sizes of the different formations: \n Shape of c: " + str(c.shape) + " \n Shape of transpose of c: " + str(d.shape) + "\n")

#In terms of shaping, reshape returns a modified version - resize actually changes the underlying inherent data to begin with
#We can also reshape in terms of running with -1
e = d.reshape(3,-1) #The -1 basically makes the elements "Shuffle" in place, as to reshape the formation to suit the parameters.
#Note, that this cannot exceed the elements given - in terms of scaling to larger dynamics (i.e, a 4*3 Matris cannot bescaled to a 20,-1, as the amount of elements is too large
print("Showcasing the result of modification in terms of reshape call with -1: \n" + str(e) + "\n")

#We can also stack different matrises
a = np.floor(10*np.random.random((2,2)))
print("Showcasing of initialization of a basic 2,2 Matris: \n" + str(a) + "\n")

b = np.floor(10*np.random.random((2,2)))
print("Showcasing of initialization of a second basic 2,2 Matris: \n" + str(b) + "\n")

c = np.vstack((a,b))
print("Showcasing a vertial stack of the two Matrises, forming a 4,2 Matris: \n" + str(c) + "\n")

d = np.hstack((a,b)) 
print("Showcasing a horizontal stack of the two Matrises, forming a 2,4 Matris: \n" + str(d) + "\n")

#If we wish, we can also stack 1D arrays into 2D arrays, with hstack
c = np.column_stack((a,b)) #Re-utilize old A and B
print("Showcasing of column stacking in terms of just smacking on B unto A: \n" + str(c) + "\n")

#If we wish, we can introduce more axises and reshape our matrises that way, if we so desire with newaxis

a = np.array([4.,2.])
b = np.array([5.,3.])
print("A 2, Matris before any modification : \n" + str(a) + "\n" + "Shape of the Matris: " + str(a.shape) + "\n")
a = a[:,newaxis]
b = b[:,newaxis]
print("We now have introduced a new Axis to the Matris: \n" + str(a) + "\n" + "Shape of the Matris: " + str(a.shape) + "\n")

#If we wish, we can further deform the Matris with introductions of even more axises and transformations
print("Showcasing of further deformation through column stacking and newaxis introductions: \n")
d = np.column_stack((a,b))
print("Before deformation: \n" + str(d) + " \n \nShape of the pre-deformation Matris: \n " + str(d.shape) + "\n") 
c = np.column_stack((a[:,newaxis],b[:,newaxis]))
print("After deformation: \n" + str(c) + " \n \n Shape of the post-deformation Matris: \n " + str(c.shape) + "\n")

#If we wish, we can specify which axis to append to in terms of calls to concatenate
print("Initialize two basic matrises of 2,2 and 1,2 \n")

a = np.array([[1,2], [3,4]])

print("Shape of a: " + str(a.shape) + "\n")
print("The first matris: \n" + str(a) + "\n")

b = np.array([[5,6]])

print("Shape of b: " + str(b.shape) + "\n")
print("The second matris: " + str(b) + "\n")

print("Showcasing of basic concatenation operations of Matrises: \n")
print(np.concatenate((a,b), axis=0))
print("\n")

#If we wish, we can further concatenate in terms of Transposes
print("Showcasing of concatenation in terms of Transposed Matrises in the concatenation call, on second Axis: \n")
print(np.concatenate((a,b.T), axis=1))
print("\n") 

#In case we wish to create specific arrays along specific Axises with range commands, we can utilize r_ and c_
print("Showcasing of utilization of r_ command to create arrays with range commands: \n")
print(np.r_[1:4,0,4])
print("\n")

#If we wish to concatenate along the second axis, we can utilize c_ for slice and concatenate
print("Showcasing of utilization of c_ command to concatenate slices along the second Axis: \n")
print("First Array is: " + str(np.array([1,2,3])) + " Second Array is: " + str(np.array([4,5,6])) + "\n")
print("Result from np.c_ on the two Arrays: \n")
print(np.c_[np.array([1,2,3]), np.array([4,5,6])])
print("\n")

#If we wish, we can split arrays in terms of horizontal splits or vertical splits
a = np.floor(10*np.random.random((2,12)))
print("Showcasing of the basic 2,12 Matris: \n" + str(a) + "\n")


#Index 1.1 -> Splits

print("================ SHOWCASING SPLITS ======================\n")
#Horizontal split of 3 parts
print("Showcasing a 3 part Horizontal split of the 2,12 Matris: \n")
print(np.hsplit(a,3))
print("\n")

print("Showcasing a 3 part Horizontal split by the 4:th Column of the 2,12 Matris: \n")
print(np.hsplit(a, (3,4)))
print("\n")

#If we wish, we could utilize Vertical splits as well, or even Array_Split to specify along which Axis to split

print("Showcasing of split of arrays: \n")
x = np.arange(8.0)
print("Basic range array: " + str(x) + "\n")
print("After split:")
print(np.array_split(x,3))
print("\n")

print("============== SHOWCASING OF SPLITS OVER ===================\n")

#Following is some showcasing of different bindings of namings and where no Copy occurs.

# Index 1.2 -> No Copies
print("========== NO COPIES OPERATIONS SECTION BEGINS ===========")
a = np.arange(12)
print("Initialize a basic array. Showcasing of the Array: \n" + str(a) + "\n")

b = a #Reassignment in terms of handle
print("Reassigned the Array to variable name b. Performing equality check of a is b: \n" + str(a is b) + "\n")

b.shape = 3,4 #Changes the shape of a
print("Reassignment of b by b.shape = 3,4 - Checking for reformation of a: \n" + str(a) + "\n")

#No copies are made either, in terms of function calls with variables

def f(a):
    print("The id of the passed variable, is: " + str(id(a)))

#Checking id of a
print("Performing manual check of id of a: " + str(id(a)) + "\n")
print("Performing function call to resolve id of a: \n")
f(a)

print("\n")
print("============ NO COPIES OPERATION SECTION ENDED =================")
print("\n")

#Index 1.3 Shallow Copy/View
print("============ VIEW OR SHALLOW COPY SECTION BEGINS ===============\n")

#If we wish to produce a view to inspect the same data, but by virtue of another object
#we can utilize views.

c = a.view() #Initialize a view of C

print("Showcasing of c: \n " + str(c) + "\n")
print("Showcasing of a: \n " + str(a) + "\n")

print("Initialized c to view the same data as a. Checking for equality: \n")
print("c is a: " + str(c is a) + "\n")

print("Performing ID check on c: ")
f(c)
print("\n")
print("Performing ID check on a: ")
f(a)

#We can also showcase that c is a view of the data owned by a
print("\nPerforming base check of c by c.base is a: " + str((c.base is a)) + "\n")

#We can access the flag data to assess the memory allocation 
print("Performing c.flags.owndata check: " + str(c.flags.owndata) + "\n")

#Even if the shape of c changes, a's does not
c.shape = 2,6
print("Showcasing c: \n" + str(c) + "\n")
print("Showcasing a: \n" + str(a) + "\n")

#However, if the data of c changes, a's data changes
c[0,4] = 5000

print("Showcasing change in data of C, causes change in a: \n" + str(c) + "\n")
print("Showcasing a: \n" + str(a) + "\n")

#Thus, the Forms are not bound - by the data is, in terms of Views

#Further more, we can showcase that slicing an array, returns a view of it

#Perform a slice of the middle of a
s = a[ : , 1:3]
print("Showcasing of the Slice: \n" + str(s) + "\n")


s[:] = 10 #Assignment of value to the Slice

print("Performing change of value in Slice, Showcasing : \n" + str(s) + "\n")
print("Showcasing of change in Slice, causes change of data in Underlying structure, since it's a view: \n" + str(a) + "\n")

print("================== VIEW OR SHALLOW COPY SECTION ENDS ===========\n")

#Index 1.4 Deep Copy

print("================ DEEP COPY SECTION BEGINS ============\n")

d = a.copy() #A new array object with new data is created

print("Showcasing the new object, d, copy of a - \n" + str(d) + "\n")
print("Performing check of equality with is, if d is a: " + str(d is a) + "\n")

print("Performing check if d bases itself on a, if d.base is a : " + str(d.base is a) + "\n")

d[0,0] = 10000
print("Showcasing of new assignment of d: \n" + str(d) + "\n")

print("Showcasing of a, to illustrate no connection/no view relationship: \n" + str(a) + "\n")
print("==================== END OF DEEP COPY SECTION =================\n")
#Further showcasing of different operations/interactions is going to be found in the upcoming segments

#In terms of NumPy, we can utilize a Concept called Broadcasting.
#Broadcasting, is the implicit conversion of operations in terms of missing elements/Axises of Arrays.

#It also consumes less memory, because implicit Broadcasting rules can circumvent allocation of greater
#resource costs. 

#Index 1.5 Broadcasting

print("======================= BROADCASTING SECTION BEGINS ==================\n")


a = np.array([1.0, 2.0, 3.0])
b = np.array([2.0, 2.0, 2.0])
c = a * b
print ("Showcasing the result of Array Multiplication: \n " +  str(c) + "\n")

#If the shapes of the Arrays implicitly match, we can utilize Broadcasting to minimize memory allocation, by Scalar multiplication instead
b = 2.0 #Re-assign b to be a Scalar instead of a Array
c = a * b #More memory efficient, due to scalar multiplication instead of Array
print("Showcasing the results of Array * Scalar Multiplication, implicit conversion is more memory efficient \n due to implicit operations. : \n" + str(c) + "\n")

#There are two rules of Broadcasting - That the dimensions are equal or that one of them is 1.

#Some examples of Broadcasting are:

#We start off by initializing a range and then reshaping it
x = np.arange(4)
xx = x.reshape(4,1)
#We create a set of trailing 1's
y = np.ones(5)
#Then we initialize a 2D Matris, 3x4, with trailing ones
z = np.ones((3,4))

#We begin by illustrating the shapes of the Matrises/Arrays
print("The shape of x is : " + str(x.shape) + "\n") #Note, it's trailing

print("The shape of y is : " + str(y.shape) + "\n") #Again, trailing

#attempting to just addition them together, will cause type errors due to shape mismatch
#However, we can come to do a different combination of shaping

print("The shape of xx is : " + str(xx.shape) + "\n") #4,1

print("The shape of xx and y, is : " + str((xx + y).shape) + "\n") #4,5 - Since 4 is aligned and the other is 1, so Broadcasting conversion works

xxy = xx +y
print("Showcasing the forming of xx + y : \n " + str(xxy) + "\n") #Broadcasting worked

print("Showcasing the shape of z : " + str(z.shape) + "\n")

xz = x + z

print("Showcasing the shape of x +z : " + str(xz.shape) + "\n") #Works because aligned dimensions in terms of 4, and 3,4

print("Showcasing of x+z : \n " + str(xz) + "\n")

#The following example showcases that we can append an axis and then result in outer product operations of arrays with it
a = np.array([0.0, 10.0, 20.0, 30.0])
b = np.array([1.0, 2.0, 3.0])

print("Showcasing the shape of a : " + str(a.shape) + "\n") #Since it is 4, and b is 3, - we can mesh the two, if we add a axis to a, converting it to 4x1
print("Showcasing the shape of b : " + str(b.shape) + "\n")

c = a[:, np.newaxis] + b   #results in a 4x3, because of mesh of 4x1 and 3,

print("Showcasing the mesh of a and b : " + str(c.shape) + "\n")
print("Showcasing the new Matris : \n " + str(c) + "\n") #the result is 1,2,3 11,12,13 21,22,23 31,32,33
#The reason for this - is because of the product of (0+1),(0+2),(0+3) (10+1)(10+2)(10+3) (20+1)(20+2)(20+3) (30+1)(30+2)(30+3)

print("================= BROADCASTING SECTION ENDS ================\n")

#In terms of NumPy, we can combine the aspects of accessing by virtue of indexes, through other Arrays

#Index 1.6 Indexing
print("================= INDEXING SECTION BEGINS ================\n")


a = np.arange(12)**2 #The first 12 square numbers

print("Showcasing the array of first 12 square numbers : \n \n " + str(a) +"\n")
i = np.array ( [ 1,1,3,8,5 ] ) #The array of indexes we will access with

b = a[i] #Access the elements by positioning of virtue of the array of i's indexes

print("Showcase the array with index accessing: " + str(b) + "\n")

#The second illustration is shaping and accessing with indexing of a 2d matris
c = np.array ( [ [ 3,4 ], [ 9, 7 ] ] )

print("Showcasing c: \n" + str(c) + "\n")
print("Showcase the shape of c: " + str(c.shape) + "\n")

print("Showcasing the accessing of a[c] : \n" + str(a[c]) + "\n")

#Showcasing of multidimensional index accessing
palette = np.array ( [ [0,0,0], #black
                            [255,0,0], #red
                            [0,255,0], #green 
                            [0,0,255], #blue
                            [255,255,255] ] ) #white
print("Showcasing the Matris being accessed: \n" + str(palette) + "\n")

image = np.array ( [ [ 0, 1, 2, 0 ], #Will assign elements to first axis
                          [ 0, 3, 4, 0 ] ] ) #Will assign elements to second axis
print("Showcasing the accessing matris: \n" + str(image) + "\n")
c = palette[image]


print("Showcasing the resulting Matris: \n" + str(c) + "\n")

#We can also access indices for different dims
a = np.arange(12).reshape(3,4)
print("Showcasing the first Matris, 3,4 - \n" + str(a) + "\n")

i = np.array( [ [ 0,1],
                  [1,2] ] ) #Indices for the first dim in terms of a
print("Showcasing the first indice matris : \n" + str(i))
j = np.array( [ [2,1],
                  [3,3] ] ) #Indices of the second dim
print("Showcasing the second indice matris : \n" + str(j))

print("Showcasing Dimension indexing in terms of a[i]: \n" + str(a[i]) + "\n")



#Remember, the shapes of the two Matrises must be the same

c = a[i,j]
print("Showcasing accessing by multi-dimensional indexing, a[i,j]: \n" + str(c) +"\n") 
#In terms of accessing, the pattern is:
# [[ a b ]]
# [[ c d ]]
#
# [[[ 0 1 2 3 ] Axis 1
#   [ 4 5 6 7 ]]  Axis 2
#
#  [[ 4 5 6 7] Axis 3
#   [ 8 9 10 11]]] Axis 4
#
# [[ a - Axis 1 -> 2  b - Axis 3 -> 5 ]] The pattern being explained here, is accessing level of axis
# [[ c - Axis 2 -> 7  d - Axis 4 -> 11 ]] Where, it first maps to axis, then Index accessing value
# Where above, i have written out the numbers of which happens to be the numerals accessed in our example

c = a[i,2]
print("Showcasing accessing with a[i,2]: \n" + str(c) + "\n")
#In this case, the mapping was just the 3:rd element of each row

#a is
# [[[ 0 1 2 3 ]
#   [ 4 5 6 7 ]]
#  [[ 4 5 6 7]
#   [ 8 9 10 11]]]
#
# j is        ##Translating for illustration
# [[2,1],   ## [[a,b],
# [3,3]]    ## [c,d]]

#                      b  a
#result is:            v  v  
#[[[ 2  1] -> [[[ 0 1 2 3 ]
#[ 3  3]]  ->             ^
#                          c , d
#                     b a
#                     v v
#[[ 6  5] ->  [ 4 5 6 7 ]] 
#[ 7  7]] ->            ^
#                         c,d
# 
#                       b   a
#                       v   v
#[[10  9]   ->  [ 8 9 10 11]]]
#[11 11]]] ->               ^
#                              c,d
c = a[:,j]
print("Showcasing accessing with a[:,j]: \n" + str(c) + "\n")

#We can also - if we wish - map i and j to a sequence (a list) and do indexing with said list

l = [i,j]
c = a[tuple(l)] #Is equivalent to doing a[i,j], however, a[i,j] formatting is deprecated - so - we have to access with tuple declaration
print("Showcasing equivalent accessing in terms of a[tuple(i,j)] : \n" + str(c) + "\n")

#We can also index arrays in terms of searching of the maximum value of time-dependent series
time = np.linspace(20, 145, 5) #Time scale
data = np.sin(np.arange(20)).reshape(5,4) #4 time-dependent series

print("Showcasing in terms of the time scale: \n" + str(time) + "\n")

print("Showcasing in terms of the time-dependent series: \n" + str(data) + "\n")

maxima = data.argmax(axis=0)
print("Showcasing of the Maxima of data: " + str(maxima) + "\n")

time_max = time[maxima] #Times corresponding to the maxima
print("Showcasing of the times of which correspond to the maxima: " + str(time_max) + "\n")

data_max = data[maxima, range(data.shape[1])] 

print("Showcasing data_max assignment: " + str(data_max) + "\n")

#We can then check co-alignment in terms of along axises
check = np.all(data_max == data.max(axis=0))
print("Showcasing checking of co-alignment of axises: " + str(check) + "\n")

#We can also enlist array values by virtue of indexing 
a = np.arange(5)
print("Showcase initialization in terms of a simple range: " + str(a) + "\n")

a[[1,3,4]] = 0 #Assign values through indexing
print("Showcase the effects of assignment through indexing: " + str(a) + "\n")

#We can also cause trailing incision in terms of repeating numerals
a = np.arange(5) #Create a range of 0-4
print("Showcasing base before re-assignment: " + str(a) + "\n")
a[[0,0,2]]=[1,2,3] #Replace the 0 with a 1, then replace the 0 with a 2 - due to repetition, the overwritten last value is 2 instead of 0
print("Showcasing of trailing numerals due to repetitious assignment: " + str(a) +"\n")

a = np.arange(5)
print("Showcasing array before modification: " + str(a) + "\n")

a[[0,0,2]] += 1 #Only 0 and 2 are added with 1
print("Showcasing array after +=1 modification: " + str(a) + "\n")

#We can also access Arrays with booleans, if we so wish - as follows
a = np.arange(12).reshape(3,4) #Make a 4x3 Matris out of a 12 long range
print("Showcasing base Matris: \n" + str(a) + "\n")

b = a > 4 #assign a boolean that is relative to a conditioning of a
print("Showcasing of b's status: \n" + str(b) + "\n")

#We can access an array that is the only overlap of between b and a, in terms of true values:
c = a[b]
print("Showcasing array that triggered with true on boolean status: \n" + str(c) + "\n")

#We can further interact with this, in terms of assignments
print("Showcasing a BEFORE modification: \n " + str(a) + "\n")
a[b] = 0
print("Showcasing a AFTER modification: \n" + str(a) + "\n")

#We can utilize boolean indexing to generate images

def mandelbrot(h,w,maxit=20):
    #Returns an image of the mandelbrot fractal of size (h,w)
    y,x = np.ogrid[-1.4:1.4:h*1j, -2:0.8:w*1j] #Returns a open mesh grid with dimension not equal to 1
    c = x+y*1j
    z = c
    
    divtime = maxit + np.zeros(z.shape, dtype=int)
    
    for i in range(maxit):
        z = z**2 + c
        diverge = z*np.conj(z) > 2**2       # who is diverging
        div_now = diverge & (divtime==maxit) #who is diverging now
        divtime[div_now] = i        #note when
        z[diverge] = 2      #avoid diverging too much
    
    return divtime
#plt.imshow(mandelbrot(400,400))
#plt.show()

#We can do a second way of indexing with booleans, which is closer to int indexing.
#For each dimension we give a 1D boolean array selecting the slices we want


a = np.arange(12).reshape(3,4)
print("Showcasing the basic 4x3 Matris first: \n" + str(a) + "\n")
b1 = np.array([False, True, True]) #first dim selection
b2 = np.array([True, False, True, False]) #The second dim selection

c = a[b1,:]
print("Showcasing selection of b1 application with a[b1,:] : \n" + str(c) + "\n")

#We can achieve the same status with just application of b1
d = a[b1]
print("Showcasing selection of b1 application with a[b1] : \n" + str(c) + "\n")

#We can also select columns in terms of the result, with b2
e = a[:,b2] #Selecting columns
print("Showcasing selection of b2 application with a[:,b2] : \n" + str(e) + "\n")

#We can, of course, also apply both sets of constraints

f = a[b1,b2]
print("Showcasing selection of b1 and b2 application with a[b1,b2] : \n" + str(f) + "\n")

#Note: The amount of dimensions must coincide in terms of lengths

#We can, if we wish - calculate vectors to account for dimensions so that these said arrays form an open mesh.

#Some examples of utilizing ix:

a = np.arange(10).reshape(2, 5)
print("Showcasing initialization of a basic 5x2 Matris : \n " + str(a) + "\n")

ixgrid = np.ix_([0,1], [2, 4])
print("Showcasing the ixgrid: \n " + str(ixgrid) + "\n")

print("Showcasing the shapes of the elements of the ixgrid : \n" + str((ixgrid[0].shape)) + "\n" + str((ixgrid[0].shape)) + "\n")

print("Showcasing the elements of the ixgrid: \n " + str((a[ixgrid])) + "\n")

#Further example showcasing computation of triplets taken from vectors
a = np.array([2,3,4,5])
b = np.array([8,5,4])
c = np.array([5,4,6,8,3])

print("Showcasing the three arrays initialized first: \n " + str(a) + "\n " + str(b) + "\n " + str(c) + "\n")

ax,bx,cx = np.ix_(a,b,c)

print("Showcasing the tuplets of each respective element: \n ax: \n " + str(ax) + "\n bx: \n" + str(bx) + "\n \n c: \n" + str(cx) + "\n")

print("Showcasing the sizes of ax, bx, cx: " + str((ax.shape)) + "\n" + str((bx.shape)) + "\n" + str((cx.shape)) + "\n")

result = ax+bx*cx
print("Showcasing the result of ax+bx*cx: \n" + str(result) + "\n")

print("Showcasing the result of [3,2,4]: \n" + str((result[3,2,4])) + "\n")

print("Showcasing the result of a[3]+b[2]*c[4]: \n" + str((a[3]+b[2]*c[4])) + "\n")

print("================= INDEXING SECTION OVER ================\n")

#Index 1.7 - Reductions

print("================= SHOWCASING REDUCTIONS ===============\n")

#We can also implement reduce function calls as follows
def ufunc_reduce(ufct, *vectors):
    vs = np.ix_(*vectors)
    r = ufct.identity
    for v in vs:
        r = ufct(r,v)
    return r

#The above version utilizes broadcasting to circumvent number of arrays needed to be created
print("Showcasing the usage of a reduction implementation with \n ix to reduce dimensions with broadcasting rules in consideration: \n " + str(ufunc_reduce(np.add,a,b,c)) + "\n")

#In terms of ufunc_reduce, we reduce the dimension of the input element by one

print("Showcasing utilization of ufunc.reduce to reduce dimensions in for instance np.multiply.reduce([2,3,5]) : \n " + str((np.multiply.reduce([2,3,5]))) + "\n")

print("Showcasing utilization of ufunc.reduce with multi-dim arrays: \n \n The 2,2,2 3d Matris : \n " + str((np.arange(8).reshape((2,2,2)))) + "\n")

print("Showcasing result of reduction: \n" + str((np.add.reduce((np.arange(8).reshape((2,2,2))), 0))) + "\n")

#In terms of axis, 0 is the default
print("Showcasing that default of axis is 0, in reduction : \n" + str((np.add.reduce((np.arange(8).reshape((2,2,2)))))) + "\n")

#We can also specify which axis to perform the reduction over
print("Showcasing the reduction over axises of whom are specified, this case, 1 : \n" + str((np.add.reduce((np.arange(8).reshape((2,2,2))),1))) + "\n")

print("Showcasing the reduction over axises of whom are specified, this case, 2 : \n " +str((np.add.reduce((np.arange(8).reshape((2,2,2))),2))) + "\n")

#We can also initialize to a specific value for initialization in terms of reduction
print("Showcasing reduction with initialization of a specific value : (np.add.reduce([10], initial=5)) : \n" + str((np.add.reduce([10], initial=5))) + "\n")

print("Showcasing reduction with initialization of a 2,2,2 Matris on axis 0 through 2, initialized to 10: \n " + str(np.add.reduce(np.ones((2,2,2)),axis=(0,2), initial=10)) + " \n")

#In terms of where we would not be able to apply reduction in terms of ufuncs without an identity, we can use reduce with initialization to np.inf
print("Showcasing of reduction with np.inf initial setting: \n " + str((np.minimum.reduce([], initial=np.inf))) + "\n")

print("================= REDUCTION SECTION OVER ================\n")

#However, attempting to perform this reduction on an empty array with no identity, causes value errors.

#Index 1.8 - Structured Arrays

print("================= STRUCTURED ARRAYS BEGINS ================\n")

#Structured arrays are ndarrays whose datatype is a composition of simpler datatypes organized as a sequence of named fields.

#Some examples

x = np.array([('Rex', 9, 81.0), ('Fido', 3, 27.0)],
                    dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])

print("Showcasing X: " + str(x) + "\n")

#The above has three fields:
#
# 'name' - a String <= 10 length 
# 'age' - 32 bit int
# 'weight' - 32 bit float

#We can access structures as per usual
print("Showcasing accessing of Structure: " + str((x[1])) + "\n")

#The usage of Structured arrays is for instance interpreting binary blobs
#
#These are intended as Structs, and for higher level data akin to tabular data
#We would be better off using Pandas in terms of circumventing poor caching
#memory behavior

#We can access fields by accessing names of said fields
x['age'] = 5
print("Showcasing X: " + str(x) + "\n")

#To account for structured arrays, we need to declare structured datatypes

#In terms of accounting for dtype, which is the datatype of the fields to account
#for declaration of length of bits - we can do as follows:

#List of Tuples
x = np.dtype([('x', 'f4'), ('y', np.float32), ('z', 'f4', (2,2))])
print("Showcasing the data type allocations: " + str((x)) + "\n")

#Further more, if we initialize as empty - we get the index assignment at that allocation
x = np.dtype([('x', 'f4'), ('', 'i4'), ('z', 'i8')])
print("Showcasing default data type allocating: " + str(x) + "\n")

#Further showcasing typing defaulting, item size and byte offset
x = np.dtype('i8,f4,S3')
print("Showcasing default data type naming allocation: " + str(x) + "\n") 

x = np.dtype('3int8, float32, (2,3)float64')
print("Showcasing default namings when type allocation is done: " + str(x) + "\n")
#The above has automatic declaration of fields that are not initialized to be something

#We can also initalize dictionaries of data types where we allocate name, format, offset, size
#Overlappings and clobbering can occur

x = np.dtype({'names' : ['col1', 'col2'], 'formats': ['i4', 'f4']})
print("Showcasing of dictionary initialization : " + str(x) + "\n")

#Second level of more specific initialization
x = np.dtype({'names' : ['col1', '<f4'],
                 'formats': ['i4', 'f4'],
                'offsets': [0, 4],
                'itemsize': 12})
#x = np.dtype({'names':['col1','col2'], 'formats':['<i4', '<f4'], 'offsets':[0,4], 'itemsize':12})

print("Showcasing of more specified Dictionary structure: \n " + str(x) + "\n")
#Overlapping cannot occur for objects due to possibility of losing pointer reference
#in terms of memory allocation when dereferencing objects

#A older version can be used in terms of < 3.6 for Python
#not used due to < 3.6 Dicts do not preserve order
# np.dtype({'col1': ('i1',0), 'col2': ('f4',1)})
#
# Would give:
# dtype([(('col1'), 'i1'), (('col2'), '>f4')])

#Showcasing some basic operations of accessing attributes 
d = np.dtype([('x', 'i8'), ('y', 'f4')])
print("Showcasing name accessing: " + str(d.names) + "\n")

#Showcasing accessing of fields
print("Showcasing field accessing: " + str(d.fields) + "\n")

#Normally, Byte offsets and Alignment are automatic declared and handled
#showcasing offsets without alignment set on

def print_offsets(d):
    print("offsets:", [d.fields[name][1] for name in d.names])
    print("itemsize:", d.itemsize)


print("Showcasing of no alignment bit trailing packaging: ")
print_offsets(np.dtype('u1,u1,i4,u1,i8,u2'))
print("\n")
#The above implication of memory handling is that the data allocation is contingent
#We can use alignment for padding to cause sometimes performance improvements
#However, most C compilers automatically do this
#IN terms of between NumPy and C, some modifications can be needed to be done on both
#ends

print("Showcasing of alignment bit packaging: ")
print_offsets(np.dtype('u1,u1,i4,u1,i8,u2', align=True))
print("\n")

#A convenience function of packing is numpy.lib.recfunctions.repack_fields which converts
#an aligned dtype or array to a packed one or vice versa

#Fields can have titles as well, as a form of alias implementation
field = np.dtype([(('my title', 'name'), 'f4')])

print("Showcasing field title allocation : " + str(field) + "\n")
#To circumvent unessecary title iteration if present, showcasing utilization of iteration

print("Showcasing iteration over names instead of redundancy of title iteration:")
for i in field.names:
    print(field.names )
    print("\n")

#The base type of structured datatypes is numpy.void
#Albeit, we can interpret them in terms of base_dtype with (base_dtype, dtype)

#Showcasing data assignment to Structured Arrays, assignment by Tuples
x = np.array([(1,2,3), (4,5,6)], dtype='i8,f4,f8')
print("Showcasing x: " + str(x) + "\n")

#Assignment
x[1] = (7,8,9)
print("Showcasing the updated x: " + str(x) + "\n")

#We can also perform assignment by Scalars
x = np.zeros(2, dtype='i8,f4,?,S1')
print("Showcasing the base x: \n" + str(x) + " " + str(x.dtype) + "\n")

#Assignment occurs to all fields
x[:] = 3
print("Showcasing updated x: \n" + str(x) + " " + str(x.dtype) + "\n")

#Assignment with a range
x[:] = np.arange(2)
print("Showcasing the updated x: \n" + str(x) + " " + str(x.dtype) + "\n")

#We can also cast Structured Arrays from one field to unstructured

#Initialize some structured arrays
twofield = np.zeros(2, dtype=[('A', 'i4'), ('B', 'i4')])

print("Showcasing the initial structured levels of arrays - 2L, shape : \n" + str(twofield) + "\n")


onefield = np.zeros(2, dtype=[('A', 'i4')])

print("Showcasing the initial structured level of array - 1L, shape : \n" + str(onefield) + "\n")

nostruct = np.zeros(2, dtype='i4')

# Won't work, because 2->Unstructured is invalid formatting requirements in terms of levels to convert 
#          V
# nostruct[:] = twofield

nostruct[:] = onefield
print("Showcasing conversion from one field, to unstructed: \n" + str(nostruct) + "\n")

#In terms of assignment from a structured array to another structured array, source -> destination
#elementwise


a = np.zeros(3, dtype=[('a', 'i8'), ('b', 'f4'), ('c', 'S3')])
print("Showcasing first structure, a: " + str(a) + "\n")

b = np.ones(3, dtype=[('x', 'f4'), ('y', 'S3'), ('z', 'O')])
print("Showcasing second structure, b: " + str(b) + "\n")

b[:] = a

print("Showcasing source overriding, b[:] = a : " + str(b) + "\n")

#When assigning to subarrays, broadcasting is involved

#Showcasing accessing individual fields by indexing the array with the field name

x = np.array([(1,2), (3,4)], dtype=[('foo', 'i8'), ('bar', 'f4')])
print("Showcasing base structure of x: " + str(x) + "\n")

print("Showcasing accessing with x['foo']: " + str((x['foo'])) + "\n") #Access first element of respective array piece, so, 1,3

#Reassigning with foo element
x['foo'] = 10
print("Showcasing after modification of foo: " + str(x) + "\n") #Modified 1,3 to 10,10

#The resulting array is a view into the original array, shares memory allocation and modification attributes there of
y = x['bar'] #Access original
y[:] = 10 #modify the view
print("Showcasing that modification of view, causes change to underlying original : " + str(x) + "\n")


#The view mostly has same dtype and itemsize
print("Showcasing typing, shape and strides: \n Type: " + str(y.dtype) + "\n Shape: " + str(y.shape) + " \n Stride: " + str(y.strides) + "\n")

#We can access multiple fields, if we wish, with a multi-field index
#However, due to updates in NumPy we have to adhere to using copy(), causing a deep copy - alloting a new variable

#Assign the base structure
a = np.zeros(3, dtype=[('a', 'i4'), ('b', 'i4'), ('c', 'f4')])

print("Showcasing the base structure Matris: " + str(a) + "\n")

b = a[['a', 'b', 'c']].copy() # Since we have to use copy() to adhere to further implementations, this is a new instance, i.e, not a view
b[['b']] = (10)
print("Showcasing the modification of the b structure, a view of a: \n" + str(b) + "\n")

print("Showcasing the base structure a, to showcase no update, as we utilized a deep copy: \n" + str(a) + "\n")

#If we access a structured array with an integer index, we get a structured scalar

x = np.array([(1,2.,3.)], dtype='i,f,f')

print("Showcasing initialization of a basic Structured Array: \n" + str(x) + "\n")

scalar = x[0] #Access the structured array with a single integer index, is a structured scalar

print("Showcasing the scalar, which is the result of accessing the structured array with a single integer: \n" + str(scalar) + "\n Typing: " + str((type(scalar))) + "\n")

scalar[0] = 10 #The structured scalar acts as a view into the original array, performing modificaiton unto it - renders changes in the underlying Array

print("Showcasing that modifying the scalar, causes changes to the original, as it's a view: \n" + str(x) + "\n")
#Note, that this inherent dynamic is unlike other NumPy scalars.

#Assign a base structured Array
x = np.array([(1,2), (3,4)], dtype=[('foo', 'i8'), ('bar', 'f4')])

print("Showcasing base structured array: " + str(x) + "\n")
#Assign the structured scalar
s = x[0]
#Modify it
s['bar'] = 100 #Since the relationship of a structured scalar is a view of the underlying array
#modifications causes changes to the original array
print("Showcasing that after modification of the structured scalar, underlying array is modified: \n" + str(x) + "\n")

#Structured scalars can also be accessed by index
scalar = np.array([(1,2.,3.)], dtype='i,f,f')[0] #Assign the array to the scalar variable
#Access the scalar by numerical index
scalar2 = scalar[0]
print("Showcasing access by index of structured scalar, scalar[0]: \n" + str(scalar2) + "\n")

#We can convert Structured scalars by calling ndarray.item
convertedTuple = scalar.item() #Convers scalar to Tuple
print("Showcasing tuple typing of Scalar by conversion with scalar.item() : \n" + str(type(convertedTuple)) + "\n")

#In order to prevent Clobbering object pointers, NumPy does not allow for viewing structured arrays containing objects

#When we perform structure comparison, in terms of equality operators - we return a equally dimensioned array
#with boolean values indicating what part is true and what is not

a = np.zeros(2, dtype=[('a', 'i4'), ('b', 'i4')])
print("Showcasing base initialization of first structure: \n" + str(a) + "\n")
b = np.ones(2, dtype=[('a', 'i4'), ('b', 'i4')])
print("Showcasing base initialization of second structure: \n" + str(b) + "\n")

c = (a == b)

print("Showcasing the equality comparison of two arrays: \n" + str(c)  + "\n")

#< and > comparisons renders false on void

#We also have a subcategory of np.arrays, that is the np record arrays
#Which has some convenience functions and interactions 

#Which means we can access by attribute instead of index

#Showcasing creating record arrays

recordarr = np.rec.array([(1,2.,'Hello'), (2,3., "World")],
                        dtype=[('foo', 'i4'), ('bar', 'f4'), ('baz', 'S10')])
print("Showcasing base record array: " + str(recordarr) + "\n")

print("Showcasing accessing by  recordarr.bar attribute in the array: " + str(recordarr.bar) + "\n")

print("Showcasing accessing by index [1:2] : " + str((recordarr[1:2])) + "\n")

print("Showcasing accessing by attribute [1:2].foo : " + str((recordarr[1:2].foo)) + "\n")

#We can also access by the reversed ordering in terms of how the indexing is written
print("Showcasing accessing by attribute recordarr.foo[1:2] : " + str((recordarr.foo[1:2])) + "\n")

print("Showcasing access by virtue of recordarr[1].baz : " + str((recordarr[1].baz)) + "\n")

#We can also convert structured arrays into record arrays
arr = np.array([(1,2.,'Hello'),(2,3.,"World")],
               dtype=[('foo', 'i4'), ('bar', 'f4'), ('baz', 'S10')])
recordarr = np.rec.array(arr)

print("Showcasing the structure before conversion: \n Contents: " + str(arr) +  "\n Typing: " + str(arr.dtype) + "\n")
print("Showcasing the structure after conversion: \n Contents: " + str(recordarr) +  "\n Typing: " + str(recordarr.dtype) + "\n")

#We can also create views of structured arrays as per follows
arr = np.array([(1,2.,'Hello'),(2,3.,"World")],
                    dtype=[('foo', 'i4'), ('bar', 'f4'), ('baz', 'a10')])

dtype = arr.dtype
print("The dtype is: " + str(dtype) + "\n")
recordarr = arr.view(dtype=dtype, type=np.recarray)

print("Showcasing view of recordarray: " + str(recordarr) + "\n")
print("Showcasing the typing of the view: " + str((recordarr.dtype)) + "\n")

#Note, the type in terms of the record array is converted into record when called in terms of a view with recarray typing

#We can perform conversion back to the plain ndarray by reseting, as is followed in a showcasing
arr2 = recordarr.view(recordarr.dtype.fields or recordarr.dtype, np.ndarray)
print("Showcasing the conversion procedure in terms of converting back to plain ndarray: \n" + str(arr2) + "\n")

#If the record array is accessed by index or attribute, they are returned as a record array if the field
#has a structured type but as a plain ndarray otherwise

recordarr = np.rec.array([('Hello', (1,2)), ("World", (3,4))],
                    dtype=[('foo', 'S6'), ('bar', [('A', int), ('B', int)])])
print("Showcasing the base record array structure: \n" + str(recordarr) + "\n")

print("Showcasing the type in recordarr's foo element: \n" + str(type(recordarr.foo)) + "\n") #Plain, no structure in indexed element

print("Showcasing the type in recordarr's bar element: \n" + str(type(recordarr.bar)) + "\n") #Record array, due to structured

#If a field has the same name as the ndarray attribute, the attribute takes presedence in terms of accessing - as of such,
#it won't be accessible by attribute - but by Index

#We can also, if we wish - drop fields, as per will be showcased

a = np.array([(1, (2, 3.0)), (4, (5, 6.0))],
    dtype=[('a', int), ('b', [('ba', float), ('bb', int)])])

print("Showcasing base initialized array with fields: \n" + str(a) + " \n" + str(a.dtype) + "\n")

a1 = rfn.drop_fields(a, 'a')
print("Showcasing array after field(a, 'a') has been dropped: \n" + str(a1) + " \n" + str(a1.dtype) + "\n")

a2 = rfn.drop_fields(a, 'ba')
print("Showcasing array after field(a, 'ba') has been dropped: \n" + str(a2) + " \n" + str(a2.dtype) + "\n")

a3 = rfn.drop_fields(a, ['ba', 'bb'])
print("Showcasing array after field(a, 'ba', 'bb') has been dropped: \n" + str(a3) + " \n" + str(a3.dtype) + "\n")

#We can also search for duplicates in a structured array along a given key
ndtype = [('a', int)]
print("Showcasing base initialization of ndtype: " + str(ndtype) + "\n")

a = np.ma.array([1, 1, 1,1,  2, 2, 3, 3],
            mask=[0, 0, 1, 0, 0, 0, 0, 1]).view(ndtype) #The overlap here is on the 3:rd and last element, causing -- outputs
print("Showcasing a: " + str(a) + "\n")

dup = rfn.find_duplicates(a, ignoremask=True, return_index=True) #Produces an array where the duplicates in terms of a is produced in an array
print("Showcasing of dup: " + str(dup) + "\n")

#If we wish, we can also get a dictionary of fields indexing lists of their parent fields
ndtype = np.dtype([('A', int),
                        ('B', [('BA', int),
                            ('BB', [('BBA', int), ('BBB', int)])])])
print("Showcasing the base initialized version of ndtype: \n" + str(ndtype) + "\n")

field = rfn.get_fieldstructure(ndtype)
print("Showcasing utilization of get_fieldstructure calling: \n" + str(field) + "\n") 
#Showcases a backwards hierarchy of parent elements
#root of chain is shown as first indexed element of the chain of elements in terms of the tree

#We can also utilize functions akin to join_by - albeit - Could not find examples for it
#The general gist of it seems to be akin to join by the designated element, malfunctions with duplicates

#We can also Merge arrays if we wish
#In case of a missing value and no mask, it will be filled with something - depending on the corresponding type
#-1 for int, -1.0 for float points, '-' for chars, '-1' for strings, True for booleans


a = np.array([1,2])
b = np.array([10., 20., 30.])
print("Showcasing of base arrays: \n a: " + str(a) + "\n b: " + str(b) + " \n") 

merge = rfn.merge_arrays((a, b))

print("Showcasing of merging of arrays: \n" + str(merge) + "\n Typing: " + str(merge.dtype) + "\n")

#If we wish, we could use masking and also merge views
mergeNoMask = rfn.merge_arrays((a,b), usemask=False)
mergeWithMask = rfn.merge_arrays((a,b), usemask=True) #Note, unless explicit call to masking is done, masking does not take place
print("Showcasing of merging with no mask: \n" + str(mergeNoMask) + "\n")
print("Showcasing of merging with mask: \n" + str(mergeNoMask) + "\n")

#Showcasing in terms of record array
mergeRec = rfn.merge_arrays(((a).view([('a', int)]),
                            b),
                            usemask=False, asrecarray=True)
print("Showcasing merging in terms of Record Array: \n" + str(mergeRec) + " \n Typing: " + str(mergeRec.dtype) + "\n")

#By virtue of merging, the typing in terms of the first element, is changed from f0 to a

#We can also add fields to existing arrays with rec_append_fields, drop fields with rec_drop_fields, join arrays r1 and r2 with
#rec_join

#We can also recursively fill fields, as showcased

a = np.array([(1, 10.), (2, 20.)], dtype=[('A', int), ('B', float)])
print("Showcasing of initialization of a: " + str(a) + "\n")
b = np.zeros((3,), dtype=a.dtype)
print("Showcasing of initialization of b: " + str(b) + "\n")

#Showcasing recursive filling from a to b
c = rfn.recursive_fill_fields(a, b)
print("Showcasing the recursively filled element of c: " + str(c) + "\n")

#We can also rename fields, if we so wish - as follows

#Initialize the np array
a = np.array([(1, (2, [3.0, 30.])), (4, (5, [6.0, 60.]))],
        dtype=[('a', int), ('b', [('ba', float), ('bb', (float,2))])])
print("Showcasing the initialized array, pre-renaming: \n Content: " + str(a) +  " \n Typing: " + str(a.dtype) + " \n ")

a = rfn.rename_fields(a, {'a':'A', 'bb':'BB'})
print("Showcasing the initialized record array after renaming: \n" + str(a) + " \n Typing: " + str(a.dtype) + " \n ")

#Further more, can we stack arrays if we so wish

#Initialize the basic array
z = np.array([('A', 1), ('B', 2)], dtype=[('A', '|S3'), ('B', float)])
print("Showcasing the first basic initialized array: \nContent:" + str(z) + " \nTyping: " + str(z.dtype) + "\n")

#Initialize the second array for stacking
zz = np.array([('a', 10., 100.), ('b', 20., 200.), ('c', 30., 300.)],
    dtype=[('A', '|S3'), ('B', float), ('C', float)])
print("Showcasing the second array used for stacking: \nContent:" + str(zz) + " \nTyping: " + str(zz.dtype) + "\n")

zzz = rfn.stack_arrays((z,zz))
print("Showcasing the stacked arrays: \nContent:" + str(zzz) + " \nTyping: " + str(zzz.dtype) + "\n")

#Since the masking does not come into play lest explicitly declared and utilized, i have chosen to not
#cover the implicit typings in terms of Masked status etc. Earlier examples showcase them, regardless.

#In terms of Reshaping, we can omit parameters to cause "Automatic" reshaping

a = np.arange(30) #Basic range
a.shape = 2,-1,3 #-1 means "whatever is needed"
print("Showcasing filling in terms of -1 element: \n Shape: " + str(a.shape) + "\n Values: \n" + str(a) + "\n")

#Showcasing illustrations of Histograms

#Initialize variance and mean
mu, sigma = 2, 0.5 #variance is 0.5^2, mean is 2
v = np.random.normal(mu,sigma,10000)

#Plot a normalised histogram with 50 bins
#plt.hist(v, bins=50, density=1)  #Create the histogram
#plt.show() #Show the histogram

#A second version we could do in terms of histograms
(n, bins) = np.histogram(v, bins=50, density=True) #NumPy version (no plot)
#plt.plot(.5*(bins[1:]+bins[:-1]), n) #Plot the points
#plt.show() #Show the plotted histogram

#We can also do some calculations as to estimate a covariance matrix, given data and weights

x = np.array([[0, 2], [1, 1], [2, 0]]) #The basic matris that we are interacting with
print("Showcasing the basic Matris, before transposition: \n" + str(x) + "\n")
x = np.array([[0, 4], [2, 2], [4, 0]]).T #Transpose the Matris, to illustrate perfect correlation, but in opposite directions
print("Showcasing the basic Matris, after transposition: \n" + str(x) + "\n")

print("Showcasing the covariance relationship: \n" + str(np.cov(x)) + "\n")

#The above relationship, is a perfect covariance, except in opposite directions, thus the reasoning for the output of the covariance matris
#We can further illustrate the covariance when combinations of elements are done
x = [-2.1, -1, 4.3] #Initialize x and y
y = [3, 1.1, 0.12]


X = np.stack((x, y), axis=0) #Stack them
print("Showcasing x,y being stacked: \n" + str(X) + "\n")
print("Showcasing their covariance matris: \n" + str(np.cov(X)) + "\n")

#We can also, if we want to - calculate the mean - specified along a said axis
#To obtain precision, we can use float64, where of float16 are used for computing float32 intermediates

#Index 1.9 -> Mean/STD/Variance

a = np.array([[0, 0], [1, 1]])
mean = np.mean(a)
print("Showcasing the mean of a basic array of 4 elements: \n" + str(a) + "\n" + str(mean) + "\n")

#We can also account for doing the calculations on different axises
c = a[0]

mean = np.mean(a, axis=0)

d = a[1]
print("Showcasing axis 0: " + str(c) + "\n")
print("Showcasing axis 1: " + str(d) + "\n")
print("Showcase the mean of axis 0: " + str(mean) + "\n")

mean = np.mean(a, axis=1)

print("Showcase the mean of axis 1: " + str(mean) + "\n")

# [a b]
# [c d]
# Axis 0 mean = (a +c)/2, (b + d)/2  :
#
# [a] [b]
#  v   v
#  +   +     /2 = Axis 0 mean
#  v   v   
# [c] [d]
################################
#
# Axis 1 mean = (a + b)/2, (c + d)/2
# [a > + > b]
#                /2 = Axis 1 mean
# [c > + > d]
# 
#

#Generally, we can reach higher accuracy with float64 in mean calculations
a = np.zeros((2, 512*512), dtype=np.float32)
print("Showcasing a: \n " + str(a) + "\n")
a[0, :] = 1.0
print("Showcasing a: \n " + str(a) + "\n")
a[1, :] = 0.1
print("Showcasing a: \n " + str(a) + "\n")

print("Showcasing the inaccuracy in terms of float32: " + str(np.mean(a)) + "\n") #Should be 0.55

print("Showcasing the more accurate float64: " + str(np.mean(a, dtype=np.float64)) + "\n") #Closer

#We can, if we wish - also implement the STD, the standard Deviation
#STD is the standard deviation, of which is the square root of the average
#of the squared deviations from the mean, i.e, std = sqrt(mean(abs(x - x.mean())**2))
#
#The average squared deviation is, normally, x.sum() / N, where N = len(x)
#
#If we specify ddof, the divisor N - ddof is used instead.
#
#In standard statistical practice, ddof=1 provides an unbiased estimator
#of the variance of the infinite pop, ddof=0 provides a maximum likelihood estimate
#of the variance for normally distributed variables.
#
#The standard deviation computed in this function is the square root of the estimated
#variance, so even if ddof=1, it won't be an unbiased estimation.
#
#In terms of complex numbers, std takes the absolute value before squaring, so that
#the result is always real and nonnegative.
#

#As for the typing, it is formulated to be the same as the input to std. Depending on
#the input, it  can cause the results to be inaccurate

a = np.array(([1,2], [3, 4])) #Initialize the basic array
print("Showcasing the basic array, a: \n" + str(a) + "\n")
print("Showcasing the standard deviation of a: " + str(np.std(a)) + "\n")
# In terms of std across axis 0
# [a b]
# [c d]
# 1 (a), 3(c) = 1 -> ((c - a)/2) Increments by 0.5 per 1 in distance between c and a
# 1 (a), 5(c) = 2  -> ((c - a)/2)
# 1 (a), 7(c) = 3 -> ((c - a)/2)
# 1 (a), 8(c) = 3.5 -> ((c - a)/2)
# 1 (a), 9(c) = 4 -> ((c - a)/2)
# 1 (a), 10(c) = 4.5 -> ((c - a)/2)
#
# The std of axis 0 is: 
# ((c - a))/2
# ((d - b))/2

#We can also specify it, to run over specific axises

b = np.std(a, axis=0)
print("Showcassing std across axis 0: " + str(b) + "\n")

c = np.std(a, axis=1)
print("Showcasing std across axis 1: " + str(c) + "\n")
# In terms of std across axis 1
# [a b]
# [c d]
# 
# abs((d - c))/2
# 3 (c), 5(d) = 1 -> ((d - c))/2 Increments by 0.5 per 1 in distance between d and c
# abs((b - a))/2
# 2 (b), 3(a) = -1, but it's absolute, so it becomes 1
# increments by 0.5 1 in distance between b and a
#
# The std of axis 1 is:
# abs((d - c))/2
# abs((b - a))/2
#
# The std of axis 0 is:
# abs((c - a))/2
# abs((d - b))/2

# [ a b ]
# [ c d ]
# The formula is std = sqrt(mean(abs(x - x.mean())**2))

#As per usual, in single precision, std() can be inaccurate
a = np.zeros((2, 512*512), dtype=np.float32)
print("Showcasing the basic matris structure: \n " + str(a) + "\n")

a[0, :] = 1.0
print("Showcasing the updated a structure: \n " + str(a) + "\n")
a[1, :] = 0.1
print("Showcasing the updated a structure: \n " + str(a) + "\n")

b = np.std(a)
print("Showcasing the inaccuracy of float32: " + str(b) + "\n")

#As per usual, we can showcase that the std in float64 is more accurate
print("Showcasing the std: " + str((np.std(a, dtype=np.float64))) + "\n")

#We can also, if we wish - handle variance computation along a specified axis
#The variance is the average of the squared deviation from the mean, i.e, var = mean(abs(x - x.mean())**2)
#
#The mean is normally calculated as x.sum() / N, where N = len(x). If, however ddof is specified, the divisor N - ddof
# is used instead. In standard statistical practice, ddof=1 provides an unbiased estimator of the variance of a 
# hypothetical infinite population.
#
# ddof=0 provides a maximum likelihood estimate of the variance for normally distributed variables.
#
# For complex numbers, the absolute value is taken before squaring, so the result is always real/non-negative

# [a b]
# [c d]
#

a = np.zeros((2, 512*512), dtype=np.float32)
print("Showcasing the base a: \n" + str(a) + "\n")

a[0, :] = 2.0 #Gives 0.2025
#1.1 gives 0.249
print("Showcasing the modified a: \n" + str(a) + "\n")
a[1, :] = 0.2
print("Showcasing the modified a: \n" + str(a) + "\n")
b = np.var(a)


#Supposedly the above is 1-0.55**2 + 0.1-0.55**2/2
#Testing some numeral exp. to see what the implementation is

#(1-0.55**2 + 0.1-0.55**2)/2
c = (1-0.55)**2 #gives 0.2025 for 1.0-0.55
d = (0.1-0.55)**2
e = (c + d)/2

#(2-1.1**2 + 0.2-1.1**2)/2
c = (2-1.1)**2 #gives 0.81
d = (0.2-1.1)**2
e = (c + d)/2

#Thus, the correct formula, for the above values is:
# 2-((2+0.2)/N)**2) + 0.2-((2+0.2)/N)**2
# Where N is the amount of elements, which in this case is 2


print("The correct answer is: " + str(b) + "\n")
print("The showcasing of e is: " + str(e) + "\n")

a = np.array([[4.0], [0.5]]) #Becomes 2-((2+0.2)/4)**2) + 0.2-((2+02)/4)**2
#a = np.array([[4.0], [0.5]]) #Try 2+1-((2+1+0.2+1)/4)**2) + 0.2+1-((2+1+0.2+1)/4)**2
print("Modified a is: \n" + str(a) + "\n")
print("The REAL var of 4.0, 0.5 is : " + str(np.var(a)))

c = (4-2.25)**2
d = (0.5-2.25)**2
e = (c + d)/2
print("The ASSUMED var of 4.0 and 0.5, is: " + str(e) + "\n" )

#As far as multi-valued Matrises in terms of into ([[1,2], [3,4]])
#I have not managed to figure out how the interaction works out in terms of Variance.

#Index 2.0 - Linear Algebra

print("===================== LINEAR ALGEBRA SECTION ======================\n")

#In terms of accounting for Cross products, we follow the following formula
# a[ x y z ]
# b[ x y z ]
# v
# c[ x y z ] -> c[ a(y)b(z) - a(z)b(y) , a(z)b(x) - a(x)b(z), a(x)b(y) - a(y)b(x) ]
#
# c(x) = a(y)b(z) - a(z)b(y)
# c(y) = a(z)b(x) - a(x)b(z)
# c(z) = a(x)b(y) - a(y)b(x)
#

#Initialize two basic 1x3 arrays of values
a = [1,2,3]
b = [4,5,6]
c = np.cross(a, b) 
# z[ (2 * 6)  - (3 * 5) , (3 * 4) - (1* 6) , (1 * 5) - (2 * 4) ]
# z[ 12 - 15, 12 - 6, 5 - 8 ]
# z[ -3, 6, -3 ] 
print("The result of the cross product is: " + str(c) + "\n")

#We also have integration in terms of broadcasting rules as is showcasted hence
a = [1,2]  #Is equivalent to [1,2,0] in our case due to broadcasting, the last spot gets padded with a 0
b = [4,5,6]
c = np.cross(a,b)
# c[ ( 2*6) - (0 * 5), (0 * 4) - (1 * 6), (1 * 5) - (2 * 4) ]
# c[ 12 - 0, 0 - 6, 5 - 8 ]
# c[ 12, -6, -3 ]
print("The result of the cross product is: " + str(c) + "\n")

#We also have the results of running with just both vectors being dimension 2
a = [1,2]
b = [4,5]
# c [ ( 2 * 0 ) - ( 0 * 5), (0 * 4) - (1 * 0), (1 * 5) - (2 * 4) ]
# c [ 0, 0, 5 - 8 ]
# c [ -3 ]
c = np.cross(a,b)
print("The result of the cross product is: \n" + str(c) + "\n")

#we can also perform multiple vector cross-products. The direction at hand of the cross
# product vector is defined by the right-hand rule.
a = np.array([[1,2,3], [4,5,6]]) #In terms of the values, inverting the order just inverts the resulting values
print("Showcasing the base structure of a, before manipulation: \n" + str(a) + "\n")
# Solve a's system first
# a [ ( 2 * 6 ) - ( 3 * 5 ), (3 * 4) - (1 * 6), (1 * 5) - (2 * 4) ]
# a [ ( 12 - 15 ), (12 - 6), (5 - 8) ]
# a [ -3, 6, -3 ]

b = np.array([[4,5,6], [1,2,3]])
print("Showcasing the base structure of b, before manipulation: \n" + str(b) + "\n")
# Solve b's system secondly
# b [ ( 5 * 3 ) - (6 * 2), (6 * 1) - (4 * 3), (4 * 2) - (5 * 1) ]
# b [ ( 15 - 12 ), ( 6 - 12 ), (8 - 5) ]
# b [ ( 3 ), ( -6 ), ( 3 ) ]

c = np.cross(a,b)
print("Showcasing the cross product of a and b: \n" + str(c) + "\n")
#c is just then a followed by b:
# array([[-3, 6, -3], 
#         [ 3, -6, 3]])
#
#Followed up Orientation with axisc keyword tommorow
#Left handed rule is to skew towards z x y
#Right handed rule is to skew towards y z x

#We can change the orientation by using the axisc keyword
c = np.cross(a, b, axisc=0)
print("Showcasing the axisc usage, changing orientation: \n" + str(c) + "\n")

#We can, if we wish - further modify the vector definition of x and y by using axisa and axisb
x = np.array([[1,2,3], [4,5,6], [7, 8, 9]])
y = np.array([[7, 8, 9], [4, 5, 6], [1,2,3]])
c = np.cross(x,y)

##In terms of accounting for Cross products, we follow the following formula
# a[ x y z ]
# b[ x y z ]
# v
# c[ x y z ] -> c[ a(y)b(z) - a(z)b(y) , a(z)b(x) - a(x)b(z), a(x)b(y) - a(y)b(x) ]
#
# c(x) = a(y)b(z) - a(z)b(y)
# c(y) = a(z)b(x) - a(x)b(z)
# c(z) = a(x)b(y) - a(y)b(x)
#
#
#x = np.array([[1,2,3], [4,5,6], [7, 8, 9]])
#y = np.array([[7, 8, 9], [4, 5, 6], [1,2,3]]) 
#
# First row:
# x[ (2 * 6) - (3*5), (3*4) - (1*6), (1*5) - (2*4) ]
# x[ 12 - 15, 12 - 6, 5 - 8 ]
# x[ -3, 6, -3 ] <- x[0] * x[1]
#
#
# x[ (5 * 9) - (6 * 8), (6 * 7) - (4 * 9), (4 * 8) - (5 * 7) ]
# x[ (45 - 48), (42 - 36), (32 - 35) ]
# x[ -3, 6, -3 ] <- x[1] * x[2]
#
# First row is: x[ -6, 12, -6 ]
#
#x = np.array([[1,2,3], [4,5,6], [7, 8, 9]])
#y = np.array([[7, 8, 9], [4, 5, 6], [1,2,3]]) 
#
# Second row:
#
# First attempt was wrong, have scrapped implementation of that one
# c[ x y z ] -> c[ a(y)b(z) - a(z)b(y) , a(z)b(x) - a(x)b(z), a(x)b(y) - a(y)b(x) ]
# 
#
# x([[a,b,c], [d,e,f], [g,h,i]])
# y([[j,k,l], [m,n,o], [p,q,r]])
#
# Second attempt, Correct
# ((x[e] * y[o]) - (x[f] * y[n]), (x[f] * y[m]) - (x[d] * y[o]), (x[d] * y[n]) - (x[e] * y[m]))
# x[ (5 * 6) - (6 * 5), (6 * 4) - (4 * 6), (4 * 5) - (5 * 4) ]
# x[ (30 - 30), (24 - 24), (20 - 20) ]
# x[ 0, 0, 0 ] <- ((x[e] * y[o]) - (x[f] * y[n]), (x[f] * y[m]) - (x[d] * y[o]), (x[d] * y[n]) - (x[e] * y[m]))
# 
# Second row becomes 0, 0, 0
#
# Third Row:
#
# y[0] * y[1] + y[1] * y[2]
# 
# y = np.array([[7, 8, 9], [4, 5, 6], [1,2,3]]) 
# a[ x y z ]
# b[ x y z ]
# v
# c[ x y z ] -> c[ a(y)b(z) - a(z)b(y) , a(z)b(x) - a(x)b(z), a(x)b(y) - a(y)b(x) ]
#
# c(x) = a(y)b(z) - a(z)b(y)
# c(y) = a(z)b(x) - a(x)b(z)
# c(z) = a(x)b(y) - a(y)b(x)
#
# y[ (8 * 6) - (9 * 5), (9 * 4) - (7 * 6), (7 * 5) - (8 * 4) ]
# y[ (48 - 45), (36 - 42), (35 - 32) ]
# y[ 3, -6, 3 ] <- y[0] * y[1]
# 
# y[ (5 * 3) - (6 * 2), (6 * 1) - (4 * 3), (4 * 2) - (5 * 1) ]
# y[ (15 - 12), (6 - 12), (8 - 5) ]
# y[ (3), (-6), (3) ] <- y[1] * [2]
#
# Third row is -> [3, -6, 3] + [3, -6, 3] 
# Third row: [6, -12, 6]
#
# Entire product is:
#
# Row one: [-6, 12, -6]
# Row two: [0, 0, 0]
# Row three: [6, -12, 6]
#

print("Showcase the structure of x before modification: \n" + str(x) + "\n")
print("Showcase the structure of y before modification: \n" + str(y) + "\n")

print("Showcasing the result of cross product: \n" + str(c) + "\n")

#We can, if we wish, change the axis definitions of the vectors as well
d = np.cross(x, y, axisa=0, axisb=0)
print("Showcasing the result of Cross product with axisa=0 and axisb=0 : \n" + str(d) + "\n")
#I could not figure out, based on rule descriptions and akin - what the interplay in terms of
#changing vector defintions of x and y, with usage of axisa and axisb, lead to actually doing.
#Thus, i have omitted the calculations and reasoning from here

#Further more, we can come to utilize the dot product of two arrays

a = np.dot(3,4)
print("Showcasing the dot product of 2 values, 3,4 : \n" + str(a) + "\n")

a = np.dot([2j, 4j], [2j, 3j])
print("Showcasing the dot product of 4 values : \n" + str(a) + "\n")
#2j * 2j = 4 * j^2 = 4 * -1 = -4
#4j * 3j = 12 * j^2 = 12 * -1 = -12
#-16 + 0j

#If we wish, we can move on to utilizing 2-d Arrays where it is the matrix product
a = [[1,0], [0,1]]
b = [[4,1], [2,2]]
c = np.dot(a, b)
print("Showcasing the dot product, where we get the Matrix product: \n" + str(c) + "\n")
# [[ a, b], [c d]]
# [[ e, f], [g h]]
# 
# [ q w ]
# [ r t  ]
#
# [ (e*a) + (b*g)  , (f*a) + (b*h) ]
# [ (d*g) + (c*e) , (c*f) + (d*h)]
#
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html#numpy.dot
#





#There is further examples that we can showcase
#Commented out prints for brevity and readability
a = np.arange(3*4*5*6).reshape((3,4,5,6)) #Is a range from 0 to 359, 6 width
#print("Showcasing the structure of first base structure: \n" + str(a) + "\n")
b = np.arange(3*4*5*6)[::-1].reshape((5,4,6,3))
#print("Showcasing the structure of the second base structure: \n" + str(b) + "\n") #Is a range from 359 to 0, 3 width

#Commit the first dot product
c = np.dot(a, b)[2,3,2,1,2,2]
print("Showcasing summation from by dot and array indexing: \n" + str(c) + "\n")
#Can also be re-written as
d = sum(a[2,3,2,:] * b[1,2,:,2])
print("Showcasing result as same: \n" + str(d) + "\n")

#We can also compute the outer product of two vectors
#The input vectors are flattened, if not already 1D

#One example is making a very coarse grid for computing a Mandelbrot set

rl = np.outer(np.ones((5,)), np.linspace(-2, 2, 5))
print("Showcasing the base structure of ones in the range of -2 to +2: \n" + str(rl) + "\n")

#Generally, the outer product is computed as follows, in general
# [ a b c ] 
# [ x
#   y
#   z ]
#   V
#   V
# [ ( x * a) , ( x * b ) , ( x * c ) ]
# [ ( y * a) , ( y * b ) , ( y * c ) ]
# [ ( z * a) , ( z * b ) , ( z * c ) ]

#                     Beginning v    v end            v denotes horizontal width of columns
im = np.outer(3j*np.linspace(2, -2, 5), np.ones((5,))) #
#                ^                         ^ denotes vertical height of columns
#            Multiplier of the j value
#Do note, the function call will attempt to "commit even splits" across the dimensions in terms of
# Width x Height

print("Showcasing the base structure of imaginary ranges :\n" +str(im) + "\n")

#To construct a grid, we can mesh together the two
grid = rl + im #Do note, the structures must have the same shape in terms of dimensions, as in 5*5 and 5*5
print("Showcasing the mesh of the real value range and the imaginary units composition: \n" + str(grid) + "\n")

#Another example, is a mesh of multiplications deriving from a source array run against outer products of numerals
x = np.array(['a', 'b', 'c', 'd'], dtype=object) #Amount of elements denotes the amounts of rows on height, so, 4 elements is 4 rows
print("Showcasing the base array: \n" + str(x) + "\n")
c = np.outer(x, [3, 1, 5]) #Amount of times each element from the base array appears in the respective row
#the result is an array with dtype of object

print("Showcasing the product of outer summing with numerals across base letter array: \n" + str(c) + "\n")



#We can also perform Singular value Decomposition, if we want
#If the input value unto svd is a 2D array, it is factorized as 
#   u @ np.diag(s) @ vh = (u * s) @ vh
#
# where u and v are 2d unitary arrays and s is a 1D of a's singular values.
#
# In case of 2D the SVD is written as A =  U S V^H, where A = a, U = u, S = \mathtt{np.diag}(s)
# and V^H = vh
#
# 
# The 1D Array s contains the singular values of a and u and vh are unitary. The rows of vh are the
# eigenvectors of A^H A and the columns of u are the eigenvectors of A A^H. In both cases, the
# corresponding (possibly non-zero) eigenvalues are given by s**2
#
# If a has more than two dimensions, then broadcasting applies.
# This means SVD is working in "stacked" mode, it iterates over all
# indices of the first a.ndim - 2 dimensions and for each combination
# SVD is applied to the last two indices.
#
# The matrix a can be reconstructed from the decomposition with either
# (u * 3[..., None, :]) @ vh or u @ (s[..., None] * vh)
#
# If a is a matrix object instead of ndarray, then so are all the return values.

a = np.random.randn(9, 6) + 1j*np.random.randn(9, 6) #A random initialized 9,6 Array combined with a Complex conjugate random structure of 9, 6
#print("Showcasing the first base initialized structure: \n" + str(a) + "\n")
print("Showcasing the shape of the first structure: \n" + str(a.shape) + "\n")

b = np.random.randn(2, 7, 8, 3) + 1j*np.random.randn(2, 7, 8, 3) #A randonly initialized 2, 7, 8, 3 Array combined with a Complex conjugate random structure of 2,7,8,3
#print("Showcasing the second base initialized structure: \n" + str(b) + "\n") Not printed for Brevity's sake
print("Showcasing the shape of the second structure: \n" + str(b.shape) + "\n")

print("========================= SHOWCASING OF LINEAR ALGEBRA SECTION OVER ====================\n")

#Index 2.1 - Allclose Interaction

print("=========================== SHOWCASING SOME ALLCLOSE INTERACTION ==============\n")

#If we wish to denote if two variables are within a tolerable relative difference, we can check that with np.allclose
#Some examples

d = np.allclose([1e10,1e-7], [1.00001e10,1e-8])
print("Showcasing result of allclose comparison of values: \n" + str(d) + "\n")

d = np.allclose([1e10,1e-8], [1.00001e10, 1e-9])
print("Showcasing result of allclose comparison of values: \n" + str(d) + "\n")

#Do note, we have to denote equal_nan as true to run into equality status of nans
d = np.allclose([1.0, np.nan], [1.0, np.nan])
print("Showcasing result of allclose comparison of nan's without equal_nan: \n" + str(d) + "\n")

d = np.allclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)
print("Showcasing result of allclose comparison of nan's with equal_nan: \n" + str(d) + "\n")

print("==================== END OF SHOWCASING ALLCLOSE INTERACTION ============\n")
#Returning to the SVD

#Index 2.2 - SVD

print(" ================== SHOWCASING SVD FOR FULL MATRISES 2D =============\n")
#We can showcase the utilization of SVD upon full matrices and shapes
u, s, vh = np.linalg.svd(a, full_matrices=True)
print("Showcasing the shapes after the svd on first structure: \n" + str(u.shape) + "\n" + str(s.shape) + "\n" + str(vh.shape) + "\n")

#d = np.allclose(a, np.dot(u[:, 6] * s, vh)) Cannot be performed due to violation of Broadcasting rules in Dimensions

smat = np.zeros((9, 6), dtype=complex) #Just a 9x6 Array of 0's with a complex conjugate typing associated with each element
print("Showcasing the base structure shape of smat: \n" + str(smat.shape) + "\n")
print("Showcasing the element being diagonally multiplied: \n" + str(s) + "\n")
smat[:6, :6] = np.diag(s)
print("Showcasing the diag incision: \n" + str(smat) + "\n") 


x = np.allclose(a, np.dot(u, np.dot(smat, vh))) 
#In terms of running tolerance check of dot product of dot product contra a
#a being a complex conjugate array structure, 9,6 shape

#Completes showcasing of reconstruction of full SVD, 2D


print("Showcasing the result of tolerance check from np.allclose(a, np.dot(u, np.dot(smat, vh))) : \n" + str(x) + "\n")

print("================= END OF SHOWCASE FOR FULL MATRIS SVD 2D ==============\n")

print("================= SHOWCASING SOME DIAGONAL STRUCTURE INTERACTION ==========\n")
#In terms of diag(), we are accessing or constructing from Diagonal resources - as can be showcased 
x = np.arange(9).reshape((3,3)) 
print("Showcasing the base range of 9, 3x3 structure: \n" + str(x) + "\n")

y = np.diag(x) #The diagonal, defaults to middle line, so 0,4,8
print("Showcasing the diagonal taken of the 3x3 structure: \n" + str(y) + "\n")

y = np.diag(x, k=1) #The k defines the diagonal line, where the numeral denotes offset from the middle index, +1 is upwards
print("Showcasing the diagonal taken of the 3x3 structure, k = +1: \n" + str(y) + "\n")

y = np.diag(x, k=-1) #The k defines the diagonal line, where the numeral denotes offset from middle index, -1 is one downwards
print("Showcasing the diagonal taken of the 3x3 structure, k = -1: \n" + str(y) + "\n")

#We can also flesh out a re-formed structure to pad the diag being extracted
y = np.diag(np.diag(x))
print("Showcasing the padded diagonal structure: \n" + str(y) + "\n")

print("============= END OF SHOWCASING DIAGONAL STRUCTURE INTERACTION =============\n")

print("================ SHOWCASING FOR REDUCED SVD OF 2D CASES ====================\n")

u, s, vh = np.linalg.svd(a, full_matrices=False)
print("Showcasing the shapes of the partial Matrises: \n u.shape: " + str(u.shape) + "\n s.shape: " + str(s.shape) + "\n vh.shape: " + str(vh.shape) + "\n")

c = np.allclose(a, np.dot(u * s, vh))
print("Showcasing the result of allclose on (a, np.dot(u * s, vh)): \n" + str(c) + "\n")

smat = np.diag(s) #The Diagonal of s, pretty straight forward
print("Showcasing the diagonal of s: \n" + str(smat) + "\n")

c = np.allclose(a, np.dot(u, np.dot(smat, vh))) #Showcasing the result of closeness comparison in terms of margin
print("Showcasing the result of closeness factor: \n" + str(c) + "\n")

print("=================== END OF SHOWCASING FOR REDUCED SVD OF 2D CASES =================\n")

#Showcasing some interaction of matmul, which we will use for the 4D Case
#The Matmul is just Matrix multiplication for most cases, of where in terms of > 2D, we run into
#the situation of taking the last two indexes and broadcasting them.
#
# It's a lot like the dot product, except Multiplication of Scalars is not allowed
# 
# and Stacks of matrices are broadcast together as if the Matrises were elements
#
# I have previously covered the pattern for matris multiplication in this Document, albeit
# I am going to cover the broadcasting rules

print("======================= SHOWCASING FOR MATMUL =======================\n")

a = np.arange(2*2*4).reshape((2,2,4)) #Range of 16 numbers, run it across shape of 2,2,4
print("Showcasing a: \n" + str(a) + "\n")
print("Showcasing shape of a: \n" + str(a.shape) + "\n") # 2 major sections, 2 subdivisions to each, 4 length of horizontal

b = np.arange(2*2*4).reshape((2,4,2)) #Range of 16 numbers, run it across shape of 2,4,2
print("Showcasing b: \n" + str(b) + "\n")
print("Showcasing shape of b: \n" + str(b.shape) + "\n") # 2 major sections, 4 subdivisions to each, 2 length of horizontal

c = np.matmul(a,b)
print("Showcasing result of matmul from np.matmul(a,b): \n" + str(c) + "\n\nShape of the result: \n" + str(c.shape) + "\n")

d = np.matmul(a,b)[0,1,1] #Is technically the same thing as sum(a[0,1,:] * b[0,:,1]) in this case
#Elementwise multiplication of a and b
# d = sum(1*4, 3*5, 5*6, 7*7) = 98
print("Showcasing the result of np.matmul(a,b)[0,1,1]: \n" + str(d) + "\n")

showcasea = a[0,1,:]
showcaseb = b[0,:,1]
print("Showcasing a[0,1,:] and b[0,:,1]: \n showcasea : " + str(showcasea) + " \n showcaseb : " + str(showcaseb) + "\n")

print("==================== END OF SHOWCASE OF MATMUL ==============================\n")

print("================== SHOWCASING FOR FULL SVD, 4D CASE ========================\n")

a = np.random.randn(9,6) + 1j*np.random.randn(9, 6)
b = np.random.randn(2, 7, 8, 3) + 1j*np.random.randn(2, 7, 8, 3)

#print("Showcasing the base structure of a: \n" + str(a) + "\n") #Not showcasing for brevity
#print("Showcasing the base structure of b: \n" + str(b) + "\n") #Not showcasing for brevity

u, s, vh = np.linalg.svd(b, full_matrices=True)
print("Showcasing the shapes of the composition: \n u's shape: " + str(u.shape) + "\n s's shape: " + str(s.shape) + "\n" + "vh's shape: " + str(vh.shape) + "\n")

x = (u[..., :3])
#print("Showcasing intermediate element of x: \n" + str(x) + "\n")

y = s[..., None, :]
#print("Showcasing intermediate element of y: \n" + str(y) + "\n")

#print("Showcasing intermediate element of vh: \n" + str(vh) + "\n")

c = np.matmul((x*y), vh) #To be put into the allclose

result = np.allclose(b, c) #Showcasing the result of comparison operations

#Another version we could do as well
secondVersion = np.allclose(b, np.matmul(u[..., :3], s[..., None] * vh))

print("Showcasing the result of operations of comparisons: \n" + str(result) + "\n")

print("Showcasing the result of the second version: \n" + str(secondVersion) + "\n")

#print("Showcasing the Matris multiplied results in comparison operations: \n" + str(c) + "\n")
#Commented out printing of the results for brevity

print("=========== END OF SHOWCASING OF FULL SVD 4D CASE ===============\n")

print("====== BEGINNING OF SHOWCASING OF REDUCED SVD, 4D CASE ==========\n")

a = np.random.randn(9,6) + 1j*np.random.randn(9, 6)
b = np.random.randn(2, 7, 8, 3) + 1j*np.random.randn(2, 7, 8, 3)

#print("Showcasing the base structure of a: \n" + str(a) + "\n") #Not showcasing for brevity
#print("Showcasing the base structure of b: \n" + str(b) + "\n") #Not showcasing for brevity

u, s, vh = np.linalg.svd(b, full_matrices=False)

print("Showcasing shapes in terms of partial matrices: \nu's shape: " + str(u.shape) + "\n" + "s's shape: " + str(s.shape) + "\nvh's shape: " + str(vh.shape) + "\n")

#The first version of closeness check
first = np.allclose(b, np.matmul(u * s[..., None, :], vh))
second = np.allclose(b, np.matmul(u, s[..., None] * vh))

print("Showcasing the results of comparisons: \nFirst comparison: " + str(first) + "\n" + "Second comparison: " + str(second) + "\n")

print("========== END OF SHOWCASING OF REDUCED SVD, 4D CASE ==========\n")

#Index 2.3 - VDot

print("========== BEGINNING OF SHOWCASING OF VDOT ==========\n")

#If we wish to get the dot product of two vectors, we can use vdot
#This takes the dot product of two vectors, flattens them to 1d Vectors.
#This should only be used on Vectors.
#
#If the first argument is complex the complex conjugate of the first argument
#is used for the calculation of the dot product

a = np.array([2+5j]) 
print("Showcasing the base structure of a: \n" + str(a) + "\n")
b = np.array([10+30]) 
print("Showcasing the base structure of b: \n" + str(b) + "\n")
#Conjugate is 2 - 5j #Basically, the conjugate is the inversed first element of the vdot product operation call
# which we apply respectively elementwise to each row

# a = np.array([2+5j])
# b = np.array([10+20])
#Conjugate is 2 - 5j
#
# 2(10 - 20) -> 20 - 40 
# +
# 5j(10 - 20) -> 50j - 100j
# -> 20 inverse(- 40 + 50j) - 100j -> 20 + 40 - 50j - 100j
# 60 - 150j

# a = np.array([2+5j])
# b = np.array([10+30])
#Conjugate is 2 - 5j
#
# 2(10 - 30) -> 20 - 60
# +
# 5j(10 - 30) -> 50j - 150j
# -> 20 inverse(- 60 + 50j) - 150j -> 20 + 60 - 200j
# 80 - 200j


result = np.vdot(a,b)
print("Showcasing the result of np.vdot(a, b): \n " + str(result) + "\n")

print("=========== SHOWCASING OF VDOT OVER ================\n")

#Index 2.4 - Choose
print("=========== SHOWCASING OF CHOOSE ==============\n")

#If we wish, we can construct a indexed array and a set of arrays to choose from

#              1 2 3, just following basic indexing
#              v v v             
choices = [0,4,2,3]
a = np.choose([2, 3, 1], choices) #Simply amasses a collections from chosen indexes across choices
print("Showcasing choose [2,3,1] from choices: \n" + str(a) + "\n")
#Result is 2,3,4

#                a                  b                       c                       d
choices = [[0, 1, 2, 3], [10, 11, 12, 13],[20, 21, 22, 23], [30, 31, 32, 33]]
c = np.choose([2, 3, 1, 0], choices) 
#Internal pointer is just incremented by 1, per element, where of it begins by index 0 of the designated index
#so, basically - c[0], d[1], b[2], a[3] - which gives us, [20, 31, 12, 3]

print("Showcasing choose [2, 3, 1, 0] from nested structure: \n" + str(c) + "\n")

#In case we wish to circumvent out of index boundary referencing, we can use clipping to
#resort to the closest edge downwards in terms of referencing
d = np.choose([2, 5, 1, 0], choices, mode='clip') # 5 clips down to the potentional maximum, which is 3
print("Showcasing choose [2, 4, 1, 0] from nested structure: \n" + str(d) + "\n")

#Then there is wrap, which basically wraps around in terms of cycling through the choices again, and lands on the
#index of which is where it would go to in the long run in terms of indexing
d = np.choose([2,4,1,0], choices, mode='wrap') 
#4 wraps around to becoming 1, due to the fact that the indexing of arrays are:
# a(0)[0, 1, 2, 3] b(1)[10, 11, 12, 13] c(2)[20,21,22,23] d(3)[30,31,32,33]
#         index 0                 index 1               index 2            index 3           Index 4 goes back to index 0, because wraps around with 1 additional step
#                                                                                                     Index 5 would be index 1, because wraps around with 2 additional steps, etc.
#                                                                                                     It's basically RESULT_INDEX = DESIGNATION_INDEX % LENGTH_OF_CHOICES
#   array index 2         array index 0                array index 1            array index 0
# d = [ 20 (c[0])    ,      1 (a[1])            ,           12 (b[2])        ,         3 (a[3]) ]
#  pointer index 0      pointer index 1               pointer index 2         pointer index 3

print("Showcasing np.choose([2,4,1,0], choices, mode='wrap') : \n" + str(d) + "\n")

#Further more, does choose utilize broadcasting, as can be showcased
a = [[1, 0, 0], [0, 1, 0], [1, 0, 1]]
print("Showcasing the base structure of a: \n" + str(a) + "\n")
choices = [-11,20] #Triggers so that on each 0, we get the first value, on a 1, we get the second indexing
c = np.choose(a, choices)
print("Showcasing choosing in terms of broadcasting rules: \n" + str(c) + "\n")

a = np.array([0, 1]).reshape((2,1,1)) #[[[0]] [[1]]]
print("Showcasing the base structure: a: \n" + str(a) + "\na's shape: " + str(a.shape) + "\n")

c1 = np.array([1,2,3]).reshape((1,3,1)) #[[[1][2][3]]]
print("Showcasing the base structure: c1: \n" + str(c1) +"\nc1's shape: " + str(c1.shape) + "\n")

c2 = np.array([-1, -2, -3, -4, -5]).reshape((1,1,5)) #[[[-1 -2 -3 -4 -5]]]
print("Showcasing the base structure: c2: \n" + str(c2) + "\nc2's shape: " + str(c2.shape) + "\n")

d = np.choose(a, (c1, c2))
print("Showcasing the result of choosing from np.choose(a, (c1, c2)), a 2x3x5 broadcasted mesh: \n" + str(d) + "\n")

#The rules of broadcasting dictate that broadening can occur if the dimensions of designation of multiplication are implicitly the same
#or 1.
#Thus, in our case of having 
# 2, 1, 1
# 1, 3, 1
# 1, 1, 5
#
# We reach that we get
# 2*1*1, 1*3*1, 1*1*5
# 2, 3, 5
# Which means 2 chunks, 3 height, 5 width
#
# Basically, the dividation is in terms of going:
# Base structure is defined by mesh combination result of a, c1, c2
# Where of values for this case, the first chunk is c1's values broadcasted to accomodate the mesh combination
# and c2 is the second chunk's values broadcasted to accomodate the mesh combination
# [[[ 1 1 1 1 1 ]
#   [ 2 2 2 2 2 ]
#   [ 3 3 3 3 3 ]]
#
# [[ -1 -2 -3 -4 -5 ]
#  [ -1 -2 -3 -4 -5 ]
#  [ -1 -2 -3 -4 -5 ]]]
#

print("================== END OF SHOWCASING CHOOSE INTERACTION =================\n")


#Index 2.5 - Compression
print("================== SHOWCASING COMPRESSION INTERACTION ==================\n")

#We can, if we wish - return selected slices of an array along a given axis

a = np.array([[1,2], [3,4], [5,6]])

print("Showcasing the base structure a: \n" + str(a) + "\n")

#Compresses along axis 0
b = np.compress([0,1], a, axis=0)

print("Showcasing the compressed version of a, along axis 0, [0,1]: \n" + str(b) + "\n")

c = np.compress([False, True, True], a, axis=0) #Deselects [1,2], but includes [3,4], [5,6]
print("Showcasing the compressed version of a with boolean indexing: \n" + str(c) + "\n")

#We can, also if we wish - select along axis 1 instead
d = np.compress([False, True], a, axis=1) #Cancels out 1, 3, 4 - but selects 2, 4, 6 given Axis 1 directioning
print("Showcasing the compressed version of a with axis 1 slicing: \n" + str(d) + "\n")

#If we were to not denote an axis, but simple work with a flattened array - we index elements, not slices
e = np.compress([False, True], a) #Cancels out 1, selects 2
print("Showcasing the indexing of the flattened array which selects elemens: \n" + str(e) + "\n")

print("=================== END OF SHOWCASE OF COMPRESSION INTERACTION ================\n")


#Index 2.6 - Cumprod/Cumsum/Inner Product
print("=================== SHOWCASE OF CUMPROD INTERACTION ===============\n")

#We can also return the cumulative product along a given axis
a = np.array([1,2,3])
print("Showcasing the basic structure: \n" + str(a) + "\n")

b = np.cumprod(a) # 1, 1*2, 2*3
print("Showcasing the cumulative product of the basic structure: \n" + str(b) + "\n")

c = np.array([[1,2,3], [4,5,6]])
print("Showcasing the basic structure initialized: \n" + str(c) + "\n")
d = np.cumprod(c, dtype=float) #We can define what type of output we wish to have as well
print("Showcasing the cumulative product of the basic structure above: \n" + str(d) + "\n")
#The results of the indexes are a cummulative multiplicative series as showcased
# 1, 1*2, 1*2*3, 4*3*2*1, 5*4*3*2*1, 6*5*4*3*2*1

#Then of course, we can run the cumulative multiplication across different columns, with axis=0
e = np.cumprod(c, axis=0) 
print("Showcasing the cumulative product across axis 0: \n" + str(e) + "\n")
# 1     ,    2       , 3
# 4*1  ,    2*5    , 6*3
# 
# Giving us
# 1,2,3
# 4, 10, 18

f = np.cumprod(c, axis=1)
print("Showcasing the cumulative product across axis 1: \n" + str(f) + "\n")
# 1 , 2*1 , 3*2*1
# 4 , 5*4 , 6*5*4
#
# 1, 2, 6
# 4, 20, 120

print("=================== END OF SHOWCASE OF CUMPROD ================\n")

print("================== SHOWCASING OF CUMSUM =================\n")

#Cumsum is cumulative sum
a = np.array([[1,2,3], [4,5,6]])
print("Showcasing the base structure: \n" + str(a) + "\n")

b = np.cumsum(a)
print("Showcasing the cumsum of the base structure: \n" + str(b) + "\n")

#[ 1 3 6 10 15 21 ]
#  1, 1+2, 1+2+3, 1+2+3+4, 1+2+3+4+5, 1+2+3+4+5+6

#We can also specify the output type, as per for instance - dtype=float
c = np.cumsum(a, dtype=float)
print("Showcasing the float typing of cumsum: \n" + str(c) + "\n")

d = np.cumsum(a, axis=0) #Run across vertical sumation
print("Showcasing base structure: \n" + str(a) + "\n")
print("Showcasing the sum across axis 0: \n" + str(d) + "\n")

# [1, 2, 3]
# [4+1, 2+5, 3+6]

# Gives [1, 2, 3]
#        [5, 7, 9]

e = np.cumsum(a, axis=1) 
# [1, 3 (2+1) , 6 (3+1+2)]
# [4, 9 (4+5) , 15 (4+5+6)]
print("Showcasing the sum across axis 1: \n" + str(e) + "\n")

print("================== END OF SHOWCASING CUMSUM ===========\n")

print("================ SHOWCASING INNER PRODUCT COMPUTATION ==========\n")

#If it is in regards to vectors (1-d Arrays) it computes the ordinary inner-product:

a = np.array([1,2,3])

b = np.array([0,1,1])
print("Showcasing base structures a and b: \n" + str(a) + "\n" + str(b) + "\n")

c = np.inner(a,b) #1*0 + 2*1 + 3*1
print("Showcasing the result from np.inner(a,b): \n" + str(c) + "\n")

#In higher dimensions, only takes the sum product of the last axes
#Given that this is a multi-dimensional structured problem, we come to use tensordot

a = np.arange(24).reshape((2,3,4))
b = np.arange(4)

c = np.inner(a,b)
print("Showcasing the inner product of multidimensional, resulting in the Tensordot product over last axises (-1, -1): \n" + str(c) + "\n")

#Another example is where the second argument b, is a scalar
d = np.inner(np.eye(2), 10) #Results in a 2x2 2D array, and a diagonal spanning with 10's. Can define k for line offset as well
print("Showcasing eye computation: \n" + str(d) + "\n")


##
print("================ END OF SHOWCASING OF INNER PRODUCT ==========\n")


##
###To help me understand how the multi dimensional problematique in terms of Tensor is computed
###We can come to utilize the tensordot product, showcased as follows:

#The original reason why i dragged up Tensordot, is due to the circumstance of that in higher dimensions,
#the operations of INner is a sum product over the last axes.

#Index 2.7 - Tensordot

print("===================== SHOWCASING OF TENSORDOT ====================\n")


#To illustrate some of the independent learning phases that i went through, on different moments,
#I have segmented some parts in commenting in regards to Tensor Products, to showcase how
#i learned and eventually figured out larger parts of the pattern

#=================================== LEARNING PHASE 1 ============================================

a = np.arange(24).reshape((2,3,4))
#b = np.arange(4)
b = np.arange(24).reshape((2,3,4))
print("Showcasing the base structures a and b: \n" + str(a) + "\n\n" + str(b) + "\n")

### a is
### [[[ 0  1  2  3]
##  [ 4  5  6  7]
##  [ 8  9 10 11]]
##
## [[12 13 14 15]
##  [16 17 18 19]
##  [20 21 22 23]]]

# b is
##['a' 'b' 'c' 'd']
#   0  1  2  3
#   4  5  6  7
#   etc. ->

## the result:
## [['bccddd' 'aaaabbbbbccccccddddddd'
##  'aaaaaaaabbbbbbbbbccccccccccddddddddddd']
## ['aaaaaaaaaaaabbbbbbbbbbbbbccccccccccccccddddddddddddddd'
##  'aaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbccccccccccccccccccddddddddddddddddddd'
##  'aaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbccccccccccccccccccccccddddddddddddddddddddddd']]
#
#

#
# [
#                               1 * b     2 * c     3 * d                 
# [0 1 2 3] -> [ [0][0] + [1][1] + [2][2] + [3][3] 
   
#                   4 * a    5 * b     6 * c     7 * d
# [4 5 6 7] -> [4][4] + [5][5] + [6][6] + [7][7] 

#                      8 * a     9 * b     10 * c       11 * d
# [8 9 10 11] -> [8][8] + [9][9] + [10][10] +  [11][11]
# ]

# [
#                            12 * a       13 * b         14 * c       15 * d
# [12 13 14 15] ->  [12][12]    [13][13]     [14][14]    [15][15]
#
#                            16 * a      17 * b        18 * c         19 * d
# [16 17 18 19] ->  [16][16]    [17][17]      [18][18]     [19][19]
#
#                           20 * a        21 * b        22 * c        23 * d
# [20 21 22 23] ->  [20][20]    [21][21]      [22][22]     [23][23] 
#
# ]

#=================================== END OF LEARNING PHASE 1 =======================

#============================= LEARNING PHASE 2 =============================


##
##a = np.arange(60.).reshape(3,4,5)
##b = np.arange(24.).reshape(4,3,2) #Base structures
##c = np.tensordot(a,b, axes=([1,0], [0,1])) #The tensor dot product
##
##print("Showcasing the base structure of Base structures (a, b) and Tensor Dot c: \na: \n" + str(a) + "\n\nb: \n" + str(b) + "\n\nc: \n" + str(c) + "\n")
##
###A equivalent way of doing this, is as can be showcased
##d = np.zeros((5,2))
##for i in range(5):
##    for j in range(2):
##        for k in range(3):
##            for n in range(4):
##                d[i,j] += a[k,n,i] * b[n,k,j]
##
###The formula for the normal Tensordot multiplication is a[k,n,i] * b[n,k,j] - where each is it's respective dimensions 
##
##print("Showcasing equality of tensor dot of np.tensordot(a,b, axes=([1,0], [0,1])) and the above:\n" + str((d == c)) + "\n")
##
###To further cement the idea of how the Tensor Product works
##a = np.array(range(1, 9))
##a.shape = (2,2,2) #Reform it into 2,2,2
##
##A = np.array(('a', 'b', 'c', 'd'), dtype=object)
##A.shape = (2,2)
##
##print("Showcasing the base structures of numerals and letters: \n" + str(a) + "\n\n" + str(A) + "\n")
##
##print("Showcasing some basic shapes of base structures:\n a's shape: " + str(a.shape) + "\n" + "A's shape: " + str(A.shape) + "\n")
##
##tensor = np.tensordot(a, A) # Third arg is default 2 for double contraction
##print("Showcasing result of tensordot in terms of letter structures: \n" + str(tensor) + "\n")
##
### [ a b ] 
### [ c d ]
###
### [[[ 1 2 ]
### [ 3 4 ]]
### [[ 5 6 ]
### [ 7 8 ]]]
###
### Gives:
### [ (a*1 + b*2 + c*3 + d*4) ]
### [ (a*5 + b*6 + c*7 + d*8) ]
###
##
##tensor = np.tensordot(a, A, 1) #Can also be specified to be 1
##print("Showcasing result of tensordot in terms of letter structures, axis 1: \n" + str(tensor) + "\n")
###
### gives [[[acc, bdd],
###           [aaacccc, bbbdddd]],
###           [[aaaaacccccc, bbbbbdddddd],
###           [aaaaaaacccccccc, bbbbbbbdddddddd]]], dtype=object)
##print("Showcasing result of tensordot in terms of shapes, in regards to axis 1: \n" + str(tensor.shape) + "\n")
##
##tensor = np.tensordot(a, A, (-1, -1))
##print("Showcasing result of tensordot in terms of shapes, in regards to axis -1, -1: \n" + str(tensor) + "\n")
### a
### v
### [ a b ]
### [ c d ]
###
##
### A
### v
### [[[ 1 2 ]
### [ 3 4 ]]
### [[ 5 6 ]
### [ 7 8 ]]]
###
### With np.tensordot(a, A, (-1, -1))
### v
### [[['abb' 'cdd']
###   ['aaabbbb' 'cccdddd']]
###
### [['aaaaabbbbbb' 'cccccdddddd']
###  ['aaaaaaabbbbbbbb' 'cccccccdddddddd']]]
###
##
### [[[ (a * 1 + b * 2) (c * 1 + d * 2) ]
###   [ (a * 3 + b * 4) (c * 3 + d * 4) ]]
###
### [[ (a * 5 + b * 6) (c * 5 + d * 6) ]
###  [ (a * 7 + b * 8) (c * 7 + d * 8) ]]]
###
### (-1, -1) indirectly can be described as
###
### [[[ ([0][0] (a * 1) + [1][1] (b * 2)) + --SPACE-- + ([2][0] (c * 1) + [3][1] (d * 2)) ]
### [ (([0][2] (a * 3) + [1][3] (b * 4)) + --SPACE-- + ([2][2] (c * 3) + [3][3] (d * 4) ]]
### 
### [[ ([0][4] (a * 5) + [1][5] (b * 6)) + --SPACE-- + ([2][4] (c * 5) + [3][5] (d * 6) ]
###  [ ([0][6] (a * 7) + [1][7] (b * 8)) + --SPACE-- + ([2][6] (c * 7) + [3][7] (d * 8) ]]]
###
###To further cement the idea of differentiating patterns of Tensor Products,
### I felt like i should illustrate more examples
##tensor = np.tensordot(a, A, (0, 1))
##print("Showcasing the result of np.tensordot(a, A, (0, 1)) : \n" + str(tensor) + "\n")
##
### [ a b ]
### [ c d ]
###
### A
### v
### [[[ 1 2 ]
###  [ 3 4 ]]
### [[ 5 6 ]
### [ 7 8 ]]]
##
####[[['abbbbb' 'cddddd']
####  ['aabbbbbb' 'ccdddddd']]
####
#### [['aaabbbbbbb' 'cccddddddd']
####  ['aaaabbbbbbbb' 'ccccdddddddd']]]
##
### [[[ (a * 1 + 5 * b) (c * 1 + 5 * d) ] 
###   [ (a * 2 + 6 * b) (c * 2 + 6 * d) ]]
###
###  [[ (a * 3 + 7 * b) (c * 3 + 7 * d) ]
###   [ (a * 4 + 8 * b) (c * 4 + 8 * c) ]]]
###
### (0, 1) indirectly can be described as 
### Index 0 of first Structure -> Index 1 of Second structure
#=========================== END OF LEARNING PHASE 2 =========================

#======================== LEARNING PHASE 3 ============================

##
### [[[ ([0][0] (1 * 0) + [1][1] (1 * 1) + --SPACE-- + ([2][0] (1 * 0) + [3][1] (1 * 1) ]
### [ (([0][2] (1*2) + [1][3] (1 * 3) + --SPACE -- + ([2][0] (1 * 0) + [3][1] (1 * 1) ]]
###
### [[ ([0][4] (1*3) + [1][5] (1 * 3) --SPACE-- + ([2][4] (1 * 3) + [3][5] (1 * 3) ]
###  [ ([0][6] (1*3) + [1][7] (1 * 3) --SPACE-- + ([2][6] (1 * 3) + [3][7] (1 * 3) ]]]
###

##
### [[[ ([0][0] (a * 1) + [1][1] (b * 2)) + --SPACE-- + ([2][0] (c * 1) + [3][1] (d * 2)) ]
### [ (([0][2] (a * 3) + [1][3] (b * 4)) + --SPACE-- + ([2][2] (c * 3) + [3][3] (d * 4) ]]
### 
### [[ ([0][4] (a * 5) + [1][5] (b * 6)) + --SPACE-- + ([2][4] (c * 5) + [3][5] (d * 6) ]
###  [ ([0][6] (a * 7) + [1][7] (b * 8)) + --SPACE-- + ([2][6] (c * 7) + [3][7] (d * 8) ]]]
##

###
#============================= LEARNING PHASE 3 ============================

#=================================== LEARNING PHASE 4 =====================================

## a is
### [[[ 0  1  2  3]
##  [ 4  5  6  7]
##  [ 8  9 10 11]]
##
## [[12 13 14 15]
##  [16 17 18 19]
##  [20 21 22 23]]]

# b is
##[[[ 0  1  2  3]
##  [ 4  5  6  7]
##  [ 8  9 10 11]]
##
## [[12 13 14 15]
##  [16 17 18 19]
##  [20 21 22 23]]]


# [[
#      a              b             0 * 0   1 * 1     2 * 2     3 * 3           -> 14      
# [0 1 2 3] * [0 1 2 3] -> [ [0][0] + [1][1] + [2][2] + [3][3] 
   
#      a              b           0 * 4    1 * 5     2 * 6   3 * 7            -> 5 + 12 + 21 -> 38
# [0 1 2 3] * [4 5 6 7] -> [0][4] + [1][5] + [2][6] + [3][7] 

#        a                 b             0 * 8    1 * 9    10 * 2     11 * 3
# [0 1 2 3]    * [8 9 10 11] -> [0][8] + [1][9] + [2][10] +  [3][11]   -> 9 + 20 + 33 -> 62
# ]

# [     a                   b              0 * 12     1 * 13    2 * 14    3 * 15
#  [0 1 2 3] * [12 13 14 15] ->  [0][12] + [1][13] + [2][14] + [3][15] ->   1 * 13 + 2 * 14 + 3 * 15 = 86
#                    
#       a                  b               0 * 16    1 * 17     2 * 18    3 * 19 
#  [0 1 2 3] * [16 17 18 19] ->  [0][16] + [1][17] + [2][18] + [3][19] ->  17 * 1 + 2 * 18 + 3 * 19 = 110
#
#       a                  b                0 * 20     21 *1     22 * 2    3 * 23
#  [0 1 2 3] * [20 21 22 23] ->   [0][20] +  [1][21] + [2][22] + [3][23] -> 21 * 1 + 2 * 22 + 3 * 23 = 134
# ]]
#


#                                                  A BLOCK
#
# [                      
#                                                 A CHUNK
#[
#     a              b               4 * 0   5 * 1   6 * 2     7 * 3   ->  5 + 12 + 21 -> 38
# [4 5 6 7] * [0 1 2 3] -> [ [0][0] + [1][1] + [2][2] + [3][3] 
#
#     a             b                 4 * 4    5 * 5    6 * 6     7 * 7
# [4 5 6 7] * [4 5 6 7] ->     [0][0] + [1][1] + [2][2] + [3][3]  ->  126
#
#     a             b                    4 * 8    5 * 9    6 * 10    7 * 11
# [4 5 6 7] * [8 9 10 11] ->   [ [0][0]   [1][1]    [2][2]     [3][3]  -> 214
# ]
#                                           
#                                           END OF CHUNK
#
#                                            A CHUNK
# [
#    a                   b               4 * 12     5 * 13      14 * 6      7 * 15 -> 302       
# [4 5 6 7] * [12 13 14 15] -> [ [0][0]  + [1][1]   +   [2][2]    +  [3][3] 
#
#   a                   b               4 * 16      5 * 17      6 * 18       7 * 19 -> 390
# [4 5 6 7] * [16 17 18 19] ->  [0][0]  + [1][1]   +   [2][2]    +  [3][3]
#
#   a                   b               4 * 20      5 * 21      6 * 22       7 * 23 -> 478
# [4 5 6 7] * [20 21 22 23] ->  [0][0]  + [1][1]    +   [2][2]    +     [3][3]
#
# ]
#                                           END OF CHUNK
#                                           
#                                           A CHUNK
# [
# ...
#  ]
#                                           END OF CHUNK
#
#] 
#                                           END OF BLOCK

# Generally, the pattern continues like this - with every "block", consisting of 3 "Chunks" ( i.e 3(row(a) * row(b) )
# iterating and performing the dot product.
#
# Of course, this does not apply to all setups of dimensions, as the chunk pattern is indirectly defined after the implicit
# dimensional forming after tensor product (i.e, 2,3,4 * 2,3,4 will not yield the same end Block pattern as say, 2,3 * 2,3
#
# This pattern is not nessecarily implicitly scaling either, in terms of a general pattern of the formations of the dimensions
# (I cannot confirm/deny this - As i have not tried. I simply wanted to understand the Tensor Product operations overall)

e = np.tensordot(a, b, axes=(-1,-1)) #Does not allow for float multiplication


print("Showcasing the tensordot product: \n" + str(e) + "\n")

#=================================== END OF LEARNING PHASE 4 =====================================

print("============ END OF SHOWCASING OF TENSORDOT PRODUCT ============\n")

#Index 2.8 - Numpy.ndarray.fill
print("============ SHOWCASING OF numpy.ndarray.fill ============\n")

a = np.array([1,2])
print("Showcasing the basic structure: \n" + str(a) + "\n")

a.fill(0) #Fill the array with 0's instead of 1 and 2's

print("Showcasing the modified structure: \n" + str(a) + "\n")

a = np.empty(2) 
#Does not actually fill the array with zeros, fills it with randomly initialzied values
#Designation of type and indexes must be done manually.
print("Showcasing emptied structure: \n" + str(a) + "\n")


a.fill(1)

print("Showcasing the filled structure: \n" + str(a) + "\n")

print("=========== END OF SHOWCASING OF numpy.ndarray.fill ==========\n")

#Index 2.9 - np.imag
print("============ SHOWCASING OF np.imag ============\n")

#We can if we so wish, return the imaginary part of the complex argument

a = np.array([1+2j, 3+4j, 5+6j])
print("Showcasing the base structure: \n" + str(a) + "\n")

b = a.imag #Basically extracts the constant before the complex arguments in an array, float format

print("Showcasing the imaginary part of the complex argument: \n" + str(b) + "\n")

#If there is complex elements in terms of the array, the returned element is imaginary
c = np.imag(1 + 1j)
print("Showcasing float extraction in terms of complex elements: \n" + str(c) + "\n")

print("=========== END OF SHOWCASING OF np.imag ===========\n")


#Index 3.0 - np.prod
print("=========== SHOWCASING OF np.prod ================\n")

#The inherent calculations in terms of prod on a 32-bit platform, is modular.
#No error is raised on overflow

result = np.array([536870910, 536870910, 536870910, 536870910 , 536870910 , 536870910 , 536870910 , 536870910])
print("Showcasing the result of numpy.prod on a 5 element array with overflow: " + str((np.prod(result))) + "\n")
#When reaching overflow, it cycles back to running into a pattern of doubling with shifting sign
# such as 16 -> -32 -> 64 -> -128 -> 256

#The product of an empty array is 1
empty = np.prod([])
print("Showcasing result of 1 from empty array computation: " + str(empty) + "\n")

#In other cases, where we are not dealing with overflow, we just sum products
theSum = np.prod([1., 2.])
print("Showcasing the result of prod: (1*2) : " + str(theSum) + "\n")

#Note, that even if the input array is 2d, we sum the product
theSum = np.prod([[1.,2.], [3., 4.]]) #1*2*3*4 -> 6*4 -> 24
print("Showcasing the result of prod: ([[1.,2.], [3., 4.]]) : " + str(theSum) + "\n")

#We can also indicate which axis to run the product over
axis1Sum = np.prod([[1.,2.], [3., 4.]], axis=1) #1*2, 3*4

print("Showcasing the result of prod over axis 1: ([[1.,2.], [3., 4.]], axis=1) :\n" + str(axis1Sum) + "\n")

#The implicit typing is adapted to being signed or unsigned in declaration, in relation to output
x = np.array([1,2,3], dtype=np.uint8) #showcasing unsigned typing implicit conversion
print("Showcasing comparison of typing: (np.prod(x).dtype == np.uint) \n" + str((np.prod(x).dtype == np.uint)) + "\n")

x = np.array([1,2,3], dtype=np.int8) #Showcasing signed implicit typing conversion
print("Showcasing comparison of typing: (np.prod(x).dtype == int) \n" + str((np.prod(x).dtype == int)) + "\n")

#We can also manually initialize the starting point of product handling, with other than 1
x = np.prod([1,2,6], initial=10) #10*2*6
print("Showcasing the result from (np.prod([1,2,6], initial=10) \n" + str(x) + "\n")

print("============ END OF SHOWCASING OF np.prod ===============\n")

#Index 3.1 - np.put
print("============ SHOWCASING OF np.put ================\n")

#We can also designate specified elementso f an array with specific values

x = np.arange(100) #Initialize a base range
print("Showcasing of x before modification: \n" + str(x) + "\n")

test = np.arange(50) #The assigned index can also be an array of indexes

np.put(x, test, [-10]) #Assign the first 50 elements with -10

print("Showcasing of x after modification: \n" + str(x) + "\n")

#We can also modify the out of boundary iterative cycling pattern with clipping mode
x = np.arange(5) #Base initialized range
print("Showcasing of x before modification: \n" + str(x) + "\n")

np.put(x, 30, -20, mode='clip') #Allows for cycling back on through the array until the amount of steps in cycling
#through has been commited.
#So, 30 steps to run over a 5 element array- is 6:th lap last element



print("Showcasing of x after put operation of: np.put(x, 30, -20, mode='clip') :\n" + str(x) + "\n")

print("=============== END OF SHOWCASE OF np.put ==========\n")


#Index 3.2 - np.putmask
print("============== SHOWCASING OF np.putmask ==============\n")

#If we wish to conditionally change elements of a structure, we can use putmask

x = np.arange(6).reshape(2,3) #Initialize the basic structure

print("Showcasing the basic initialized structure: \n" + str(x) + "\n")

#       base structure
#             v   v- condition of triggering operation upon each respective element being iterated over
np.putmask(x, x>1, x**2) 
#                       ^ operation and assignment of new value after operation completes

#putmask is in-place working on the working memory, so you cannot perform
#assignment, as the function itself returns none.

#To circumvent this, we just call the function on the structure instead of performing assignment

print("Showcasing x after modification: \n" + str(x) + "\n")

print("=============== END OF SHOWCASING OF np.putmask ==============\n")


#Index 3.3 - numpy.real
print("=============== SHOWCASING OF numpy.real ===============\n")

a = np.array([1+2j, 3+4j, 5+6j])

print("Showcasing the base structure: \n" + str(a) + "\n")

b = a.real

print("Showcasing the real part of complex arguments: \n" + str(b) + "\n")

a.real = 20 #Re-designate the real part of the complex arguments

print("Showcasing the re-designated real parts of complex arguments: \n" + str(a) + "\n")

#We can also designate each respective real element with an array
a.real = np.array([9,8,7])

print("Showcasing of real part designation with array, in relation to complex arguments: \n" + str(a) + "\n")

#We can also access the real component directly
a = np.real(1 + 1j)

print("Showcasing direct accessing of real components of complex arguments: \n" + str(a) + "\n")

print("======================= END OF SHOWCASING OF numpy.real ==============\n")

#Index 3.4 - numpy.sum

print("================ SHOWCASING OF numpy.sum =================\n")

#We can, if we wish - run sum of array elements over a given axis
#Akin to before, the Arithmetic does not raise an error upon overflow

#Showcasing summing of an empty array
suming = np.sum([])
print("Showcasing the result of sum on a empty array: \n" + str(suming) + "\n")

#Showcasing some basic interactions
suming = np.sum([0.5, 1.5]) #1.5 + 0.5
print("Showcasing the result of basic sumation: \n" + str(suming) + "\n")

#Showcasing forced typing and rounding implicit conversion
suming = np.sum([0.5, 0.7, 0.2, 1.5], dtype=np.int32) #Forcing int typing causes rounding down, meaning 0 (0.5) + 0 (0.7) + 0 (0.2) + 1 (1.5)

print("Showcasing rounding down in case of forced int typing: \n" + str(suming) + "\n")

#Showcasing basic sumations, again
suming = np.sum([[0,1], [0,5]]) # 0 + 1 + 0 + 5 = 6

print("Showcasing the suming: \n" + str(suming) + "\n")

#Showcasing both axis 0 and axis 1 sumations

suming = np.sum([[0, 1], [0,5]], axis=0) #0 + 0, 1 + 5 = 0, 6

print("Showcasing the suming in terms of axis 0: \n" + str(suming) + "\n")

suming = np.sum([[0, 1], [0,5]], axis=1) #0 + 1, 0 + 5 = 1, 5

print("Showcasing the suming in terms of axis 1: \n" + str(suming) + "\n")

#A interesting notion is that if an accumulator is too small, overflow occurs
#Basically this means that, if you run with a too restrictive of a typing in terms of data allocation max size
#overflow occurs when you go beyond the limit of said typing
suming = np.ones(129, dtype=np.int8).sum(dtype=np.int8) #Overflows, due to too small Max size container of data type

print("Showcasing the result of suming overflow due to too small datatype containment: \n" + str(suming) + "\n")


#We can also start the sumation with something else than zero

suming = np.sum([0.5, 1.5], initial=10) #10 + 0.5 + 1.5 -> 12
print("Showcasing the sumation with other than start of 0: \n" + str(suming) + "\n")

print("==================== END OF SHOWCASING OF numpy.sum ====================\n")

#Index 3.5 - numpy.argmax

print("==================== SHOWCASING OF numpy.argmax ====================\n")

#Argmax is meant to return the indices of the maximum value along a axis

#initialize the base structure
a = np.arange(6).reshape(2,3)

a[0][1] = 20
a[0][0] = 30
a[1][0] = 44 #Access second array, initialize first value to 6

print("Showcasing of a: \n" + str(a) + "\n")

#                                 INDEX   0 1 2    3 4 5
maxofa = np.argmax(a) #3, since   [[30,20,2], [44,4,5]]


print("Showcasing of argmax of a: \n" + str((np.argmax(a))) + "\n")


#The interesting part occurs when we denote axis
maxofa = np.argmax(a, axis=0) 
#Gives index on the vertical comparison
# 1, 0, 1
# 44 > 30, 20 > 4, 5 > 2

print("Showcasing of argmax of a with axis 0: \n" + str(maxofa) + "\n")

#where as of, if we run with axis 1, we end up with the index across the horizontal
maxofa = np.argmax(a, axis=1) #0, 0 -> 
#                                      30 > 20 and 30 > 2, so, 0
#                                      44 > 4 and 44 > 5


print("Showcasing of argmax of a with axis 1: \n" + str(maxofa) + "\n")

#We can also aquire the maximal elements in terms of a N-dimensional array
ind = np.unravel_index(np.argmax(a, axis=None), a.shape)

print("Showcasing the indexing of accessing max values from the base structure: \n" + str(ind) + "\n")

#We can then get back the maximum by virtue of accessing with the indexing retrieved
maxing = a[ind]

print("Showcasing retrievel of maximum through a[ind]: \n" + str(maxing) + "\n")

#As for the maximal element in terms of indices, only the first hit of hte max value is returned, upon duplicates found
a = np.arange(6)
print("Showcasing basic structure of a range structure: \n" + str(a) + "\n")

a[1] = 5 #Assignment

print("Showcasing modified basic structure: \n" + str(a) + "\n")

c = np.argmax(a) #Retrieve the index of max of where there is 2 5's, only first indice for 5 will be returned, 1
print("Showcasing index retrieval of max upon duplicates found, only first is retrieved: \n" + str(c) + "\n")

print("==================== SHOWCASING OF numpy.argmax OVER ====================\n")

#Index 3.6 - np.unravel_index

print("=========================== SHOWCASING OF UNRAVEL_INDEX ======================\n")

#I wish to denote how unravel_index works

a = np.arange(12).reshape(3, 4)

print("Showcasing base structure of a 3*4, to illustrate the unravel better: \n" + str(a) + "\n")

unravel = np.unravel_index(6, (3, 4)) #Find the index of 6, in a range structure of 3x4

#A 3x4 is a basic range, so:
#             > COLUMN
# V ROWS  | 0 | 1 | 2 | 3   
#-----------------------
#              0 | 1 | 2 | 3      | Row nr. 0
# ----------------------
#              4 | 5 | 6 | 7      | Row nr. 1
#-----------------------
#            8 |    9     |    10    |    11     | Row nr. 2
#       Col 1    Col 2      Col 3     Col 4

#unravel_index gives us 1,2 -> Access row 1, column 2
#so, 6 is accessed by 1,2

print("Showcasing unravel_index(6, (3, 4)): \n" + str(unravel) + "\n")

b = np.arange(25).reshape(5, 5)

print("Showcasing the base structure of b: \n" + str(b) + "\n")

unravel = np.unravel_index(10, (5, 5)) #Find the index of 10, in a 5x5 range structure
#2,0
#Because:
# 0 1 2 3 4
# 5 6 7 8 9
# 10 11 12 13 14 <- 2,0 the index of 10
# 15 16 17 18 19
# 20 21 22 23 24

print("Showcasing accessing of np.unravel_index(10, (5, 5)): \n" + str(unravel) + "\n")

#We can, if we wish - unravel a more complicated pattern in terms of accessing with arrays with corresponding indexes of unravelling
c = np.arange(42).reshape(7,6)
print("Showcasing the base 7,6 system we are unravelling: \n" + str(c) + "\n")
unravel = np.unravel_index([22, 41, 37], (7, 6)) #Showcase a 7,6 system unravelling
print("Showcasing a 7.6 unravel system call with np.unravel_index([22, 41, 37]) : \n" + str(unravel) + "\n")

#gives (array([3, 6, 6], dtype=int64), array([4, 5, 1], dtype=int64))
#Basically, we find:
#22 at 3,4
#41 at 6,5
#37 at 6,1

#We can also denote what type of indexing should be categorized according to row major or column major with
# order indication (F for FORTRAN or C for C)

print("====================== END OF SHOWCASING OF UNRAVEL_INDEX =================\n")

#Index 3.7 - numpy.argmin

print("===================== SHOWCASING OF numpy.argmin ========================\n")

#The correspondant to argmax, is argmin

a = np.arange(6).reshape(2,3)

print("Showcasing basic structure: \n" + str(a) + "\n")

#Extract index of the smallest element
c = np.argmin(a)

print("Showcasing of the index of the smallest element: \n" + str(c) + "\n")

#When running comparison, we will get 0,0,0 - for every element in a range in terms of first row,
#is less than the second ones elements
c = np.argmin(a, axis=0)

# [0 1 2]
# [3 4 5]

print("Showcasing comparison in terms of argmin comparison: \n" + str(c) + "\n")
#0 < 3, 1 < 4, 2 < 5 - Thus, the smallest index on each comparison is the first, in our case giving 0,0,0

#As far as minimum running across a axis of 1
c = np.argmin(a, axis=1)

print("Showcasing comparison in terms of argmin across axis 1: \n" + str(c) + "\n")
#The smallest element in a rage segmentation with [0,1,2] [3,4,5]
#Gives that 0 < 1 and 2, 3 < 4 and 5
#Thus, 0,0 are the indexes of the smallest elements

#Where as of this time in terms of unravelling of indexes, we find that 0,0 is the main source
ind = np.unravel_index(np.argmin(a, axis=None), a.shape) #Will give 0,0 - as that is the source of the smallest point in a range that grows from 0
# 0, 1, 2
# 3 4 5
#
# Thus, 0,0 yields the smallest - as it's 0
print("Showcasing of unravelling index in terms of argmin accessing: \n" + str(ind) + "\n")

#And we can of course, backtrack to 0 with the accessing of the same unravelling indexing
print("Showcasing the backtracking to 0 through accessing of ind parsing, with a[ind]: \n" + str((a[ind])) + "\n")

#Repeating showcasing of return of only the first index in terms of duplicates being present
b = np.arange(6)

print("Showcasing the basic range: \n" + str(b) + "\n")

b[4] = 0

print("Showcasing the modified range: \n" + str(b) + "\n")

#Illustrating that accessing with argmin against duplications leaves to indexing of 0 again
c = np.argmin(b) #Will assign 0, as the occurence of the first 0 is at index 0

print("Showcasing of the argmin index accessing: \n" + str(c) + "\n")

print("========================= END OF SHOWCASING OF ARGMIN ========================\n")

#Index 3.8 - np.argsort

print("========================= SHOWCASING OF argsort =============================\n")

#In terms of argsort, it returns the indices that would sort an array

#1d Arrays
x = np.array([3, 1, 2])
argsort = np.argsort(x) 
#The argsort will return the indexes in terms of if the array was sorted
#                       
#In this case, it will be [1, 2, 0] - as this indirectly would correspond to [1,2,3]
print("Showcasing the returned indexes of argsort: \n" + str(argsort) + "\n")


#Showcasing the case of two-dimensional array interactions
x = np.array([[5, 1], [1,10]])
print("Showcasing the base structure of a 2d Array: \n" + str(x) + "\n")

#We can sort along axises as showcased
c = np.argsort(x, axis=0) 
print("Showcasing the result of sorting along axis 0: \n" + str(c) + "\n") #Sorting of basis of downwards along axis, vertical
#
# [[5, 1], [1,10]] will give
# [1 1] 1 , 1
# [0 0] 5, 10
#
# In case we had [[1,1], [5,10]] instead
# we would have gotten
# [0 0] 1, 1
# [1 1] 5, 10

#In case of [[5,1], [1,10]]
# we would have gotten
# [1 0] 1, 1
# [0 1] 5, 10

#In terms of Horizontal, we can do as follows:
d = np.argsort(x, axis=1)
#Would give 
# [1, 0] -> reflects 1, 5
# [0, 1] -> reflects 1, 10

#in case of [[1,5], [10,1]]
# would give
# [0,1] -> reflects 1,5
# [1,0] -> reflects 1,10

#As far as indices go, we can unravel with indexes as follows

print("Showcasing the result of sorting along axis1: \n" + str(d) + "\n")

print("Showcasing structure of x before unravel_index call: \n" + str(x) + "\n")

ind = np.unravel_index(np.argsort(x, axis=None), x.shape) 
#Since it has no designation of algorithmics in sorting,
#we end up with a situation of where a merged array with each respective indexing
print("Showcasing of the result of unraveling_index of x: \n" + str(ind) + "\n") 

#(array([0, 1, 0, 1], dtype=int64), array([1, 0, 0, 1], dtype=int64))
#Gives this, because it sorts to [1,1,5,10]
#thus
# [ [0][1] (1), [1][0] (1), [0][0] (5), [1][1] (10) ]
#

#We can also sort by virtue of accessing keys
x = np.array([(1,0), (0,1)], dtype=[('x', '<i4'), ('y', '<i4')])

print("Showcasing the base structure: \n" + str(x) + "\nTyping: " + str(x.dtype) + "\n")

c = np.argsort(x, order=('x', 'y'))
print("Showcasing the structure after np.argsort(x, order=('x', 'y')): \n" + str(c) + "\nTyping: " + str(c.dtype) + "\n")

c = np.argsort(x, order=('y', 'x'))
print("Showcasing the structure after np.argsort(x, order=('y', 'x')): \n" + str(c) + "\n Typing: " + str(c.dtype) + "\n")

print("============================== END OF SHOWCASING OF argsort =====================\n")

#Index 3.9 - numpy.ptp

print("============================== SHOWCASING OF numpy.ptp =========================\n")

#Gives the sum of the range of two values between designated axises, as follows

# [[0, 1]
#  [2, 3]]

x = np.arange(4).reshape((2,2))
print("Showcasing the base structure of x: \n" + str(x) + "\n")

#Ptp stands for "peak to peak", but it's basically maximum - minimum in terms of each respective axis end
a = np.ptp(x, axis=0) 
#Vertical, since axis=0:
# 2 - 0, 3 - 1
# 2, 2

print("Showcasing of np.ptp(x, axis=0): \n" + str(a) + "\n")

a = np.ptp(x, axis=1)
#Horizontal, since axis=1:
# 1 - 0, 3 - 2
# 1, 1

print("Showcasing of np.ptp(x, axis=1): \n" + str(a) + "\n")

print("=========================== END OF SHOWCASING OF PTP =======================\n")

#Index 4.0 - numpy.searchsorted

print("=========================== SHOWCASING OF numpy.searchsorted =================\n")

#We can, if we wish - denote to find the indexes of where elements should be inserted to maintain order

insertionIndex = np.searchsorted([1,2,3,4,5], 3) #To have a ordered array with the insertion of three, we have to put it after 2, thus, index 2

print("Showcasing index result of np.searchsorted([1,2,3,4,5], 3): \n" + str(insertionIndex) + "\n")

#We can also, if we wish - denote alignments in terms of directions of where the element is to coincide with
insertionIndex = np.searchsorted([1,2,3,4,5], 3, side='right') #Right side alignment, but keep the ordering result after insertion
print("Showcasing index result of np.searchsorted([1,2,3,4,5], 3): \n" + str(insertionIndex) + "\n")

#We can also denote insertion for an entire array, of where we find respective indexes elementwise computed and denoted
insertionIndex = np.searchsorted([1,2,3,4,5], [-10, 10, 2, 3])
# 
# The result is [0 5 1 2], since it can account for the positive/negative value on a horizontal axis to account for insertion indexes
print("Showcasing index result of np.searchsorted([1,2,3,4,5], [-10, 10, 2, 3]): \n" + str(insertionIndex) + "\n")

print("============================ SHOWCASING OF np.searchsorted OVER ======================\n")

#Index 4.1 - np.sort

print("============================ SHOWCASING OF np.sort =============================\n")

#We can, if we wish - perform sorting operations in terms of returning sorted copies of arrays

#In general, it can be denoted to be said that there is 3 different algorithms to use
#
# kind          speed       worst case          work space          stable
#
# *quicksort     1           O(n^2)              0                       no
# * Quicksort was changed to be a switch off unto heapsort if not enough progress is done. Which causes O(n*log(n)) worst case runtime
#
# mergesort    2           O(n*log(n))        -n/2                   yes
#
# heapsort      3           O(n*log(n))         0                       no
#

# The general mechanics in terms of sort algorithmics, is that they make temp compies
# of the data when sorting along any but the last axis.
#
# Consequently, it's fastest/uses the least space to sort along the last one.
#
# The ordering order is real > complex, except when the real == complex,
# where the complex value defines the order.
#
# Stable mode automates to mergesort due to stability of solution integration

#In terms of nan handling, the following showcases how ordering is computed
# 
# Real: [R, nan] Where R is a real non-complex value
# Complex: [R + Rj, R + nanj, nan + Rj, nan + nanj]

#Some examples showcasing the interactions of sorting
smallBase = np.array([[1,4], [3,1]])

print("Showcasing base structure before sort interactions: \n" + str(smallBase) + "\n")

a = np.sort(smallBase) #Sorts along hte last axis, thus
#1,4 
#1,3

print("Showcasing the base structure after sort interactions: \n" + str(a) + "\n")

bigBase = np.array([[20,3], [10,5,8, 15], [60,60,60]])

print("Showcasing the big structure before sort interactions: \n" + str(bigBase) + "\n")

b = np.sort(bigBase)
print("Showcasing the result of larger structure with last axis base sort: \n" + str(b) + "\n")


#In terms of parsing with no axis as input, we flatten the array in the interactions
flatA = np.sort(a, axis=None)

print("Showcasing axis=none sort call on small base structure. Flattens structure: \n" + str(flatA) + "\n")

#To sort along the first axis, i.e horizontal, we can do axis 0
a = np.sort(smallBase, axis=0)
print("Showcasing of sorting along first axis: \n" + str(a) + "\n")

#If we wish, we can further denote ordering by type, structure and other factors.
#As can be showcased as follows:

#Begin with denoting typing in an array
dtype = [('name', 'S10'), ('height', float), ('shoesize', int),('age', int)] #acts akin to a dictionary-ish in terms of structure

print("Base structure is: \n Name: \n Height: \n Shoesize: \n Age: \n")

values = [('Dick', 2.556668, 30, 400), ('Joe', 2.556668, 35, 400), ('Flanders', 2.3, 44, 450)] #Initializing with identical values on two
secondValues = [('Xavius', 2.556668, 35, 400),('Abra', 2.556668, 35, 400), ('Joe', 2.556668, 35, 400), ('Flanders', 2.556668, 35, 400)]
#elements to find out how comparison in terms of sub-sectioning of ordering occurs

#Create a structured array which meshes together typing and values
a = np.array(values, dtype=dtype)  #Meshes together the two arrays, binding values to typings

a2 = np.array(secondValues, dtype=dtype)

sortedA = np.sort(a, order='height')
sortedA2 = np.sort(a2, order='height')

print("Showcasing of ordering by order of height: \n" + str(sortedA) + "\n")
#The difference between these two, showcases that implicit sub typing after values are identical, can be delegated to things like
#name attribute and akin
print("Showcasing ordering by alphabetical in case of identical cases otherwise: \n\n" + str(sortedA2) + "\n")

#We can specify this ordering, as to denote what order of subsectioning should occur, in terms of 'equal' cases
sortedA = np.sort(a, order=['height','age', 'shoesize', 'name'])
sortedA2 = np.sort(a2, order=['height','age', 'shoesize', 'name'])

print("Showcasing of order by order of height > age > shoesize > name: \n\n" + str(sortedA) + "\n\n" + str(sortedA2) + "\n\n")

print("======================= SHOWCASING OF np.sort DONE ======================\n")

#Index 4.2 - np.all

print("======================= SHOWCASING OF np.all =============================\n")

#We can run operations of evaluations in terms of arrays of elements, such as to see if they evaluate to true

#Items that are not 0, are deemed as True. This includes NaN and Infinities

#Initialize the base structure
a = [[True, False], [True, True]] #Contains a False, so will compute to False

result = np.all(a)
print("Showcasing question operation on base tuple [[True, False], [True, True]] : \n" + str(result) + "\n")

#We can evaluate along axises as well, such as axis 0 or axis 1
result = np.all(a, axis=0) #Run check on the vertical (0 is vertical, 1 is horizontal)

print("Showcasing a question operation on base tuple [[True, False], [True, True]], with axis 0 : \n" + str(result) + "\n")

#Gives True, False - because True, True and False True
# [[True, False]
#  [True, True]]
#    ^       ^      False + True = False, since one is False
#    ^
# True + True = True, since all are true
# 

#We can of course, run the questions operations on numericals as well
base = [-1, 4, 5]
result = np.all(base) #Since all are != 0, this computes to true

print("Showcasing result from np.all([-1, 4,5]): \n" + str(result) + "\n")

base = [1.0, np.nan] #True, since != 0 and a nan
result = np.all(base) 

print("Showcasing result from np.all([1.0, np.nan]): \n" + str(result) + "\n")

#We can also Alternative the output array to parse the results to, with the keyword out

#I tried a number of different operations in relation to the out argument. I kept getting an error
#of mismatch of dimensions, despite checking shapes of the different structures and attempting to
#run with the examples displayed in the documentation.
#
#As of such, i have excluded the showcasing of the keyword usage of out.

print("========================= END OF SHOWCASE OF np.all ==================\n")

#Index 4.3 - numpy.any

print("========================= SHOWCASING numpy.any ======================\n")

#Checks for wether any value in an array is true or not, according to previous guidelines

base = np.array([[True, False], [True, True]]) #Flattened array contains a true value, 3 to be precise

result = np.any(base)

print("Showcasing of result, performing np.any(base) call on [[True, False], [True, True]] : \n" + str(result) + "\n")

#We can run it along an axis, as well, if we wish
base = np.array([[True, False], [False, False]])

result = np.any(base, axis=0) 
#Run comparison along Vertical, giving : [[True, False] [False, False]]
# structure composition, to which gives us:
# [True, False] from np.any([True, False], axis=0)

print("Showcasing of result, performing np.any(base) call on [[True, False], [False, False]] : \n" + str(result) + "\n")

#Of course, we can run evaluations along numerical arrays as well
base = np.array([-1, 0, 5]) #Initialize base numerical basic structure
result = np.any(base) #True, since -1 and 5 are != 0, thus, 2 trues, any results True

print("Showcasing of result, performing np.any(base) call on [-1, 0, 5] : \n" + str(result) + "\n")

base = np.array(np.nan) #Initialize basic nan element in a array structure
result = np.any(base) #True, since nan is not equated to 0

print("Showcasing of result, performing np.any(base) call on [np.nan] : \n" + str(result) + "\n")

##o = np.array([False])
##z = np.any([-1, 4, 5], out=o)
##
##print("Showcasing of z: \n" + str(z) + "\n")

#The traceback in terms of attempting to utilize the out argument is:
#
#Traceback (most recent call last):
##  File "...\sci1.py", line 3284, in <module>
##    z = np.any([-1, 4, 5], out=o)
##  File "...\numpy\core\fromnumeric.py", line 2013, in any
##    return _wrapreduction(a, np.logical_or, 'any', axis, None, out, keepdims=keepdims)
##  File "...\numpy\core\fromnumeric.py", line 83, in _wrapreduction
##    return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
##ValueError: output parameter for reduction operation logical_or has too many dimensions
#
# It seems to be a inherent problem, albeit, i have chosen to just bypassingly denote that it exists here,
# instead of dragging it out and attempting to circumvent the issue

print("========================= END OF SHOWCASING numpy.any ======================\n")

#Index 4.4 - numpy.nonzero

print("========================= SHOWCASING numpy.nonzero ======================\n")

#If we wish to find out the indices of an array where the elements are non-zero, we can use numpy.nonzero

#Initialize the base structure
x = np.array([[1,0,0], [0,2,0], [1,1,0]])

print("Showcasing the base structure: \n" + str(x) + "\n")

result = np.nonzero(x) 
#The result is two arrays, each containing mapping indexes to each
#respective index of where we find elements referencing
#             0:th index array      1:st index array        2:nd index array
# Base is : [       [1,0,0],              [0,2,0]           , [1,1,0]       ]
# Reference indexes are:
#
# [0, 1, 2, 2] Because 0:th array, 1:th array and 2:nd array twice
# [0, 1, 0, 1] Because 1, 2 and 1,1 are non-zero elements

print("Showcasing the result of calling np.nonzero(x): \n" + str(result) + "\n")

#We can of course, transpose the structure as well - if we so desire
transposedResult = np.transpose(np.nonzero(x))

print("Showcasing the result of transposing the structure: \n" + str(transposedResult) + "\n")

#We can run similarity in terms of conditioning checks in terms of Boolean arrays
# in tandem with nonzero calls - due to False being interpreted as 0
a = np.array([[1,2,3],[4,5,6],[7,8,9]]) #The base structure

print("Showcase the base structure: \n" + str(a) + "\n")

checkGreater = a > 3

print("Showcase the boolean structure array of condition a > 3: \n" + str(checkGreater) + "\n")

#We can now utilize the boolean array to run operations of nonzero to run evaluations on it

evaluations = np.nonzero(checkGreater) #Run non-zero indexing check on boolean array

print("Showcasing the result of running nonzero on the boolean array: \n" + str(evaluations) + "\n")
#Because the entire first row is 0's, it'll trigger for indexes of the rest of the stuff

#Inherently, boolean arrays carry the method of nonzero() which acts in a similar fashion
evaluations2 = ((a > 3).nonzero())

print("Showcasing the result of running nonzero on the boolean array with .nonzero(): \n" + str(evaluations2) + "\n")

print("========================= SHOWCASING OF numpy.nonzero() OVER =====================\n")

#Index 4.5 - numpy.where()

print("========================= SHOWCASING OF numpy.where() ===========================\n")

#We can showcase the utilization of where in tandem with booleans and numeral indexes, as follows

result = np.where([[True, False], [False, False]], #Based on False or not - it defines which array we pick
                      [[30, 2], [3, 4]],
                      [[9, 8], [7, 6]])

#In case of False index denoted, we run with the second, True is first one
# Thus:
# [[False, False], [False, False]]
# 
# on
#
# [[5, 2], [3,4]]
# [[9, 8], [7,6]]
#
# Gives ->
# [[9, 8]
#  [7 6]]
#
# Which is roughly following the pattern of
# 
# if True -> a, else -> e
#   V        
#   V if True -> b, else f
#   V V
#   V V   if True -> c, else g
#   V V   V 
#   V V   V  v if True -> d, else h
# [[a, b], [c, d]] 
#
# [[e, f],  [g, h]]
# 

print("Showacsing the result of running np.where check in tandem with boolean array: \n" + str(result) + "\n")

#We can of course, run with heavier restrictions of conditions as well
x = np.arange(9.).reshape(3, 3)
print("Showcasing the base structure: \n" + str(x) + "\n")

result = np.where(x > 5) #Just illustrating basic conditioning comparison
print("Showcasing the check in terms of seeing where elements are of what stature: \n" + str(result) + "\n")

#The returned result is the two arrays, first being binding of index of array, the second of index of element

#Do note, running a conditionof where it is never triggered, gives two empty arrays

result = x[np.where(x > 3.0)] #Basically accesses values of the indexings accessed, so we convert to 1d with a array of values of the indices returned

print("Showcasing the check in terms of seeing where elements can be sub-sectioned into accessing: \n" + str(result) + "\n")


#Do denote, that if we run with insufficient elements to cover the base array being checked against,
#we can run with broadcasting to fill in blanks with a designated value:

result = np.where(x < 5, x, "N/A")

print("Showcasing the broadcasting of filling in rest elements: \n" + str(result) + "\n")

result = np.where(x < 5, x, 500)

print("Showcasing implicit typing conversion in terms of output: \n" + str(result) + "\n")

#Note the difference between the two - as implicit typing type casts the inherent results based on input of argument.
#As of such, converting rest to String, causes the rest to be string implicitly converted as well.
#
#Same goes for Numerics.

#We can also utilize methods akin to isin to produce boolean arrays and then run comparisons
base = [3,4,2] #Initialize basic structure

print("Showcasing of base structure: \n" + str(base) + "\n\n")
result = np.isin(x, base) #Run basic check of comparison of values

print("Showcasing of looking for base values in the x structure: \n" + str(result) + "\n")

#We can run a where check on this, of course, but it's basically just converting booleans to indexes at this point
indexes = np.where(result) #Goes from boolean structure, to index structure

print("Showcasing of conversion in terms of indexes contra boolean structure: \n" + str(indexes) + "\n")

print("========================= SHOWCASING OF numpy.where() ENDED ===========================\n")

#Index 4.6 - numpy.array_split()

print("========================= SHOWCASING OF numpy.array_split ===========================\n")

#If we wish to split arrays, we can do so by virtue of Array_split
#How this splits, is that it takes the closest denominator it can fit in terms of a even split
#and then distirbutes iteratively from left to right with the remainder

x = np.arange(9) #Initialize basic range
print("Showcasing of the basic range structure: \n" + str(x) + "\n")

split = np.array_split(x, 4)
#9/4 = 2.5
# 2 is the whole, so each respective slice is 2 indexes
# 4 * 2 = 8, 9 - 8 = 1 -> Thus, one extra element gets shuffeled unto the first slice

#[0,1,2] [3,4] [5,6] [7,8]

print("Showcasing of split command unto range of 9 structure: \n" + str(split) + "\n")

x = np.arange(30) #Bigger initial range

print("Showcasing of the basic range structure: \n" + str(x) + "\n")

split = np.array_split(x, 7)
#30/7 = (4*7) + 2
#Thus, 7 slices of length 4, with the first 2 being length 5

print("Showcasing of the split command unto range of 30 structure: \n" + str(split) + "\n")

print("========================= SHOWCASING OF numpy.array_split ENDED ===========================\n")

#Index 4.7 - numpy.column_stack

print("========================= SHOWCASING OF numpy.column_stack =============================\n")

#If we wish, we can stack columns by virtue of taking a sequence of 1-D arrays and stacking them as columns
#to make a single 2-D array. 2-D arrays are stacked - 1D's are converted to 2D columns.

#Initialize the base structure
a = np.array((1,2,3))
b = np.array((2,3,4))

#Stack the columns
stack = np.column_stack((a,b))
print("Showcasing of the stack result: \n" + str(stack) + "\n")

print("Further illustrating shape of the resulting stack: \n" + str((stack.shape)) + "\n")
#The above is to illustrate that the arrays have been converted to a 2D stacked structure of 2 columns

print("========================= SHOWCASING OF numpy.column_stack DONE =============================\n")

#Index 4.8 - numpy.concatenate

print("========================= SHOWCASING OF numpy.concatenate ==============================\n")

#If we wish to run concatenation operations on arrays, we can do so

a = np.array([[1, 2], [3, 4]])
print("Showcasing base structure of a: \n" + str(a) + "\n")
b = np.array([[5, 6]])

print("Showcasing base structure of b: \n" + str(b) + "\n")

#Showcasing the concatenated status of the two
c = np.concatenate((a,b), axis=0) #Just concatenate on the vertical, merging the two

print("Showcasing the result of concatenation: \n" + str(c) + "\n")

#We can of course concatenate along axises and transpose, etc.
d = np.concatenate((a, b.T), axis=1) #The result of a and transposed B structure concatenated on the axis1
#Axis 1 is horizontal

print("Showcasing the basic transpose of b: \n" + str((b.T)) + "\n") #Since the transpose is basically just flipping axises and alignments
#We end up with adapting to account for the horizontal instead of vertical with a transpose

print("Showcasing the concatenation in terms of axis 1: \n" + str(d) + "\n") #b's elements get concatenated to each respective row
#as the base is 
# [1,2] + 5
# [3,4]  + 6

#Where as of running concatenation along no axis, causes merge unto one element
f = np.concatenate((a,b), axis=None)

print("Showcasing the result of no axis concatenation operation: \n" + str(f) + "\n")

#We can of course, run into interactions in terms of Masking
a = np.ma.arange(3) #The masked range 

print("Show the basic structure of the masked range structure: \n" + str(a) + "\n")

a[1] = np.ma.masked

print("Showcasing a after it being modified: \n" + str(a) + "\n")

b = np.arange(2, 5) #Initialize a basic range

print("Showcase a basic range: \n" + str(b) + "\n")

#We then just concatenate the two to showcase basic interactions
d = np.concatenate([a, b]) #Does not preserve masking

print("Showcasing the result of basic concatenation: \n" + str(d) + "\n")

d = np.ma.concatenate([a,b]) #Does preserve masking
print("Showcasing the result of running np.ma.concatenate: \n" + str(d) + "\n")

print("=========================== SHOWCASING OF np.concatenate/np.ma.concatenate OVER ===============\n")

#Index 4.9 - numpy.diagonal

print("=========================== SHOWCASING OF numpy.diagonal ===============\n")

#We can access diagonals if we so wish, albeit due to versionings conflictings and akin - we end up in a situation
#of where we wish to access a copy to then later be able to modify it and use it etc.
#
#The reasoning being, that diagonal() returns a read only copy

a = np.arange(4).reshape(2,2)

print("Showcasing of base structure: \n" + str(a) + "\n")

theDiagonal = np.diagonal(a).copy() #By accessing a copy, we get something we can write to 
test = a.diagonal()


print("Showcasing of the base structure copy, can be modified: \n" + str((theDiagonal)) + "\n") #Can be modified, is a copy

print("Showcasing of the diagonal in no copy mode, which is read only: \n" + str(test) + "\n") #Cannot be modified, is read only

#We can also off-set the k diagonal line by index (positive is up, negative is down)
showcaseDiag = a.diagonal(1)

print("Showcasing the diagonal with offset of 1 in upwards direction, in terms of k: \n" + str(showcaseDiag) + "\n")

showcaseDiag = a.diagonal(-1) #Showcasing interaction of minus line direction implication

print("Showcasing the diagonal with offset of -1 in downwards direction, in terms of k: \n" + str(showcaseDiag) + "\n")

#Where as of, we can do the same dynamics for 3d of course
a = np.arange(8).reshape(2,2,2); 

#depending on the shape of the Matris, defines how the diagonal is taken.
#In the case of 2,2,4 - it's the First row in 2,2,4 - placed on the vertical
#whilst the lowest row is on the vertical for the second row

## 2,2,4 Shape, no diag index
##[[[ a  b  c  d]
##  [ e  f  g  h]]
##
## [[ i  j k l]
##  [m n o p]]]

# Gives, in terms of Diagonal
#
# [[ a m ]
#  [ b n ]
#  [ c o ]
#  [ d p ]]

## 2,2,2 Shape, no diag index
##[[[ a b ]
##  [ c d ]]
##
## [[ e f ]
##  [ g h ]]]

# Gives, in terms of Diagonal
# 
# [[ a g ]
#  [ b h ]]

print("Showcasing the base structure of the 3D structure: \n" + str(a) + "\n")

diagA = a.diagonal() #Just take the basic diagonal first to showcase it

print("Showcasing the basic diagonal taken on the 3d Structure: \n" + str(diagA) + "\n")

#If we wish, we can access each specific row with indexing, which showcases which diagonals we were accessing in the first place
firstDiag = a[:,:,0]
#                           [[ a c ]
#                           [ e g ]]
print("Showcasing first row taken diagonal from: \n" + str(firstDiag) + "\n")

secondDiag = a[:,:,1] 
#                          [[ b d ]
#                          [ f  h ]]

print("Showcasing second row taken diagonal from: \n" + str(secondDiag) + "\n")


print("========================= SHOWCASING OF numpy.diagonal ENDED ==============\n")

#Index 5.0 - numpy.dsplit

print("========================= SHOWCASING OF numpy.dsplit ================\n")

#If we wish, we can split arrays into smaller pieces - with parameters for categorization and 
#factorization in terms of subpartitionings
#

x = np.arange(16.0).reshape(2, 2, 4) #Initialize a basic range structure and reshape it to be 2,2,4

print("Showcasing the basic 2,2,4 range of 16 structure: \n" + str(x) + "\n")

splitbytwo = np.dsplit(x, 2) #Just splits it on the middle and segregates them to two different arrays, basically

print("Showcasing the split example: \n\n" + str(splitbytwo) + "\n")

#In case of no parameter being fed in, we just split and segregate by the last axis, where the remainder gets put as scraps
x = np.arange(30).reshape(5,3,2)

print("Showcasing the basic structure of 5,3 range of 15 structure: \n" + str(x) + "\n")

d = np.dsplit(x, np.array([3,6]))

print("Showcasing dsplit in terms of 30 range, into 3,6: \n" + str(d) + "\n") #Generally, remnant pieces are just relegated
#to be empty arrays in terms of after the split

print("========================= SHOWCASING OF numpy.dsplit DONE ================\n")

#Index 5.1 - numpy.dstack

print("========================= SHOWCASING OF numpy.dstack ==============\n")
#We can, also - if we wish - stack based on depth along the third axis - This is similar to division by dsplit, as it
#basically rebuilds thoose from that state

#Initialize basic structures
a = np.array((1,2,3))
b = np.array((2,3,4))

#Basically gives
# [[[ 1, 2 ],
#   [ 2, 3],
#   [ 3, 4]]] 
#


c = np.dstack((a,b))

print("Showcasing in terms of stack result: \n" + str(c) + "\n")

#We can, of course - also run with array indexing stacks
a = np.array([[1], [2], [3]]) 
b = np.array([[2], [3], [4]]) 

c = np.dstack((a,b))
#Sort a vertically and shuffle it to the left
#next, sort b vertically, shuffle to the next index, etc.

print("Showcasing in terms of stack result: \n" + str(c) + "\n")

print("===================== END OF SHOWCASING numpy.dstack ==============\n")

#Index 5.2 - numpy.hsplit

print("===================== SHOWCASING OF numpy.hsplit ===================\n")

#Where as of dsplit was based on axis=2, hsplit is the same principle, but based on axis=1

base = np.arange(16.0).reshape(4,4)

print("Showcasing the base structure: \n" + str(base) + "\n")

basesplit = np.hsplit(base, 2) #Split it before index 2, once

print("Showcasing the split by index 2: \n" + str(basesplit) + "\n")

basesplit = np.hsplit(base, np.array([1,3])) #We can split by several indexes if we so wish

print("Showcasing the split by index 1 and 3: \n" + str(basesplit) + "\n")

#even if the case of the base is multi dimensional, we will end up splitting across the second axis

base = np.arange(16.0).reshape(4, 2, 2)

print("Showcasing in terms of the base structure: \n" + str(base) + "\n")

basesplit = np.hsplit(base, 2) #Note that the relationship between the base shape and the split must be even subdivisionary
#lest a exception is thrown

print("Showcasing in terms of the basesplit: \n" + str(basesplit) + "\n")

#Basically the split takes the first upper column of the section and concatenates it with the next chain strung together
#
# [a b c]
# [d e f]
#
# [g h i]
# [j k l]
#
# Gives in our case
#
# [a b c] -> 0 1 2
# [g h i] -> 6 7 8
#
# [d e f] -> 3 4 5
# [j k l] -> 9 10 11

print("===================== SHOWCASING OF numpy.hsplit OVER ===================\n")

#Index 5.3 - numpy.hstack

print("===================== SHOWCASING OF numpy.hstack ========================\n")

#We can stack arrays horizontally - which rebuilds the arrays from the horizontal splits

a = np.array((1,2,3))
b = np.array((2,3,4))

print("Showcasing the basic base range structure: \n" + str(a) + "\n" + str(b) + "\n")

horizontalstack = np.hstack((a,b))

print("Showcasing the horizontal stack: \n" + str(horizontalstack) + "\n")

#We can, of course - do it with basic array pieces if we wish
#But when done in a horizontal matter with this, we get the structure of actual horizontal piece by piece
#instead of the entire arrays being concatenated one on the end of the other
a = np.array([[1], [2], [3]])
b = np.array([[2], [3], [4]])

horizontalstack = np.hstack((a,b)) #The horizontal stack 

print("Showcasing the horizontal stack: \n" + str(horizontalstack) + "\n")

print("=================== SHOWCASING OF numpy.hstack OVER ===============\n")

#Index 5.4 - numpy.ndarray.item

print("=================== SHOWCASING OF numpy.ndarray.item ===============\n")

#We can access copies of a specific element at specific indexes if we so wish

base = np.random.randint(9, size=(3, 3))

print("Showcasing the base random intialized structure: \n" + str(base) + "\n")

#For instance, we can access items and copy then by index as showcased
item = base.item(3) #Access item at index 3

#We can, if we wish - also denote to access by row and index
item = base.item(2,2) #Same as treating the structure as an array with [2][2]

print("Showcasing the accessing of a row and a index: \n" + str(item) + "\n")

print("====================== SHOWCASING OF ndarray.item OVER ===========================\n")

#Index 5.5 - numpy.newaxis

print("====================== SHOWCASING OF numpy.newaxis =============================\n")

#Newaxis is an alias for None, which we can utilize to index arrays

status = newaxis

print("Showcasing newaxis: \n" + str(status) + "\n")

print("Showcasing comparison to see if newaxis is None: \n" + str((None == newaxis)) + "\n")

#Initialize a base to insert a empty space into it with newaxis
base = np.arange(3)

print("Showcasing the basic base range: \n" + str(base) + "\n\nShape: " + str(base.shape) + "\n")

base = base[:, newaxis] #Create a copy, except with a insertion of an empty space

print("Showcase the shape after having inserted the newaxis: \n" + str(base.shape) + "\n")

base = base[:, newaxis]

print("Showcase the shape after having done a second insertion of the newaxis: \n" + str(base.shape) + "\n")

print("Showcasing the basic modified element: \n" + str(base) + "\n")

#We can, if we so wish - make multiple insertions of newaxises as we can showcase
base = base[:, newaxis, newaxis]

print("Showcase the shape after having done a double insertion of the latest shape: \n" + str(base) + "\n\n" + str(base.shape) + "\n")

#We can also utilize broadcasting in tandem of multiplying it with a greater shape, to showcase dimension interplays

secondBase = np.arange(15).reshape(3, 5)

base = base[:, newaxis] * secondBase #If we were to exclude the newaxis, we would get a different result

test = base[:] * secondBase #No newaxis in multiplication

print("Showcase the result without having included newaxis: \n" + str(test) + "\n\n" + str(test.shape) + "\n")
#Retains same shape in terms of multiplication result

print("Showcase the base having been multiplied by itself: \n" + str(base) + "\n")

print("Showcase the new shape: \n" + str(base.shape) + "\n") #Note how when we multiply the two, we find that 
#it goes from 3,1,1,1,1
# to 3,1,1,1,3,5 in terms of Shape

print("=================== SHOWCASING OF numpy.newaxis OVER ===============\n")

#Index 5.6 - numpy.ravel

print("=================== SHOWCASING OF numpy.ravel ====================\n")

#If we wish, we can, "ravel" an input array containing the elements of the input
#A copy is made only if needed

#Showcasing some of the interactions in terms of ravel
base = np.array([[1,2,3], [4,5,6]])

print("Showcasing the base structure: \n" + str(base) + "\n")

ravelbase = np.ravel(base) #This is technically the same thing as base.reshape(-1)

print("Showcasing the raveled base structure: \n" + str(ravelbase) + "\n")

#We can of course, ordane different ordering formats if we so wish
ordering = np.ravel(base, order='F') #Basically sort to 1,4,2,5,3,6
# [[ a b c], [ d e f ]]
# [[ a, d, b, e, c, f ]]

print("Showcasing the ordering in terms of order of F: \n" + str(ordering) + "\n")

#To ilustrate A ordering, we need to first initialize a transpose
transpose = base.T

print("Showcasing the transposed base: \n" + str(transpose) + "\n")

test = np.ravel(transpose)

print("Showcasing the transposed base raveled: \n" + str(test) + "\n")

#We will get back to the base order of F, if we call it to sort to A, in the above state

test = np.ravel(transpose, order='A')

print("Showcasing that the transposed in terms of Order of A, keeps F ordering: \n" + str(test) + "\n")

#K will read them as per storage in memory, except when reversing the data when strides are negative

newBase = np.arange(3) 

print("Showcasing the new base structure: \n" + str(newBase) + "\n")

reversedNewBase = newBase[::-1] #Reverse ordering of the base structure

print("Showcasing the intermediate step of reversing: \n" + str(reversedNewBase) + "\n")

#Basically, C will read elementwise in terms of the column inputs,
# [a b c] -> a, b, c
# [d e f] -> d, e, f etc.
# regardless of value in terms of memory.

#However, K, will preserve ordering and not swap axeses

kBase = np.arange(12).reshape(2,3,2) #To be able to showcase the base structure

print("Showcase the base of K, which we will operate upon: \n" + str(kBase) + "\n")

kShowing = kBase.swapaxes(1,2) #Just swap the axes of of 1 to 2, so go from Horizontal iterative
# To horizontal based on vertical
#Basically shuffle the angle, so that we go from:
# [ a b ]
# [ c d ]
# [ e f ]
# 
# [ g h ]
# [ i j ]
# [ k l ]

# to 
#
# [ a c e ]
# [ b d f ]
#
# [ g i k ]
# [ h j l ]

print("Showcase the kShowing variable: \n" + str(kShowing) + "\n")

#If we  then ravel that, we get either C ordering, as in swapped axises of 1,2
#or we get iterative listed as the base shape

cRavel = kShowing.ravel(order='C')
print("Showcasing the cravel result from swapped axises: \n" + str(cRavel) + "\n")

kRavel = kShowing.ravel(order='K')
print("Showcasing the kravel result from swapped axises: \n" + str(kRavel) + "\n")

#The order of F is the fortran-styling indexing - where column index sorts the slowest
#whilst index runs the quickest

print("====================== SHOWCASING OF np.ravel OVER ===================\n")

#Index 5.7 - np.repeat

print("====================== SHOWCASING OF np.repeat ===================\n")

#We can run into repeating across different axises, if we so wish
base = np.repeat(3, 4) #Base structure ot showcase 4 indexes of 3

print("Showcasing np.repeat(3,4) : \n" + str(base) + "\n")

base = np.array([[1,2], [3,4]])

baseRepeated = np.repeat(base, 2) #repeat every element so they appear twice

print("Showcasing the base: \n" + str(base) + "\n\n Showcasing the repeat from base: \n" + str(baseRepeated) + "\n\n Shape: " + str(baseRepeated.shape) + "\n")
#Where as of, if we divident in terms of axises - we add another dimension

baseRepeatedAxis1 = np.repeat(base, 5, axis=1) #Run repeat of 5 of each element, add another dimension to the shape
print("Showcasing after axis call and shape, to illustrate with axis call: \n" + str(baseRepeatedAxis1) + "\n\n Shape:" + str(baseRepeatedAxis1.shape) + "\n")

baseRepeatedAxis2 = np.repeat(base, [10, 10], axis=0) #Denote on what direction the axis is to expand unto
# Each respective index in the array denotes how many times the designated row is to repeat.
#
# So 10,10 - means - repeat each row [1 2] and [3 4] 10 times, on the Vertical (axis 0)
#
# Thus, we get
# [1 2]
# [1 2]
# ... v  Continues for a total of 10 times
# [3 4]
# [3 4] 
# ... v Continues for a total of 10 times

print("Showcasing np.repeat(base, [10, 10], axis=0) : \n" + str(baseRepeatedAxis2) + "\n")

print("========================= SHOWCASING OF np.repeat OVER =====================\n")

#Index 5.8 - np.reshape

print("========================= SHOWCASING OF numpy.reshape ======================\n")

#We can of course, reshape structures without changing the underlying data - if we so wish

#Initialize the base structure
a = np.zeros((10, 2))

print("Showcasing in terms of the base structure: \n" + str(a) + "\n")

#b = a.T re-assign the transpose to a temporary variable
#c = b.view() create a view of it, re-assign

#c.shape = (20) #Shape defaults and implies to C orientation in terms of sorting and adherence
#Causing contigious memory allocation and referencing/parsing to be a thing of a restriction
#
#As in, attempts to reshape outside of the inherent allocations of how the base structure is,
#attempting to re-ordane to F style, as in, based on switching columns - that would break
#memory allocations.
#
#Thus, attempts of re-ordination here, would raise an error - it would break memory formation

#We can, thus, logically - also ordane wether we should implement what level of Ordering,
#based on the above dynamics. As in, we can implement order of C or F, in terms of Reshape.
#
# Where of, the inherent dynamics of reshape is based on akin to raveling and hten reshaping, basically

#We can of course, showcase the dynamics of reshaping, as can be shown
a = np.array([[1,2,3], [4,5,6]]) #Basic shape initialization
print("Showcasing the inherent base structure: \n" + str(a) + "\n")
reshaped = np.reshape(a, 6) #Reshapes the structure unto one structure
print("Showcasing of the structure after modification: \n" + str(reshaped) + "\n")

#Of course, we can further re-herse the inherent dynamics of reshaping and integrations
showcase = np.reshape(a, 6, order='F') #Resort to Fortram style assorting
print("Showcasing basic result of ordering by Fortran re-structuring: \n" + str(showcase) + "\n\n Shape:" + str(showcase.shape) + "\n")

showcase = np.reshape(a, (2, -1))
print("Showcasing the result of reshaping in terms of implicit formation with -1: \n" + str(showcase) + "\n Shape:" + str(showcase.shape) + "\n")

print("============================ SHOWCASING OF RESHAPE OVER =========================\n")

#Index 5.9 - np.resize

print("============================= SHOWCASING OF RESIZE ===============================\n")

#If we wish to resize, we can do so. 
#Denote that the interactions of resize is different of np.resize compared to <base element>.resize(<Columns>, <Rows>)

a = np.array([[0,1], [2,3]]) #Initialize base structure
print("Showcasing the base structure: \n" + str(a) + "\n")

a = np.resize(a, (5,3)) #Designates rows and width of each row 
print("Showcasing the resized version: \n" + str(a) + "\n")

d = np.array([[1,4], [5,6]]) #Initialize a structure to help illustrate for <base element>.resize(<Columns>, <Rows>)
e = d.resize(10, 10) #To denote difference in how elements are filled out.
#As in, filling out the rest with 0's, unlike np.resize, which copies elements in terms of filling

print("Showcasing the difference in <base element>.resize contra np.resize(columns, rows): \n" + str(e) + "\n")

print("================================= SHOWCASING OF RESIZE OVER =================\n")

#Index 6.0 - numpy.squeeze

print("================================= SHOWCASING OF numpy.squeeze =================\n")

#We can utilize this to remove the Single-dimensional entires from the shape of an array

base = np.array([[[0], [1], [2]]]) #Initialize the base structure

print("Showcasing the shape of the base structure: \n" + str(base.shape) + "\n")

squeezedBase = np.squeeze(base) #remove the 1D elements of the shape

print("Showcasing the squeezed structure: \n" + str(squeezedBase.shape) + "\n")

#Where as of, we can also designate to squeeze across specific axises as well
squeezedAxisZero = np.squeeze(base, axis=0) #Run squeeze across Axis 0
# On a shape of 1,3,1
#                   ^ Is removed, on axis 0, leaving us with
# 3,1 as a shape

print("Showcasing the squeezed along axis 0 structure shape: \n" + str(squeezedAxisZero.shape) + "\n")

#Where of, if we tried to squeeze an axis which does not have a size of 1, we get an error
#
# 1,3,1
#    ^ Cannot perform a squeeze on the 3, it's not a 1D index 

#squeezedAxisOne = np.squeeze(base, axis=1) #Will run into an error

#Where of, we can squeeze along axis 2, as can be showcased
squeezedAxisTwo = np.squeeze(base, axis=2)
# On a shape of 1,3,1 
#                        ^ Is removed, on axis 2, leaving us with
# 1,3 as a shape 

print("Showcasing the squeezed along axis 2 structure shape: \n" + str(squeezedAxisTwo.shape) + "\n")

print("==================== SHOWCASING OF numpy.squeeze OVER =================\n")

#Index 6.1 - numpy.swapaxes

print("==================== SHOWCASING OF numpy.swapaxes =====================\n")

#We can run operations of swapping axises as well, if we wish
base = np.array([[1,2,3], [4,5,6]])

print("Showcase base structure: \n" + str(base) + "\n")

#                                  V<<<<<<<<<<< Go from Horizontal, 0
#                                  V  v<<<<<<<<<< To Vertical, 1                       
swapAxes = np.swapaxes(base,0,1) #Basically flips the orientation unto vertical, from horizontal

print("Showcase the swapped axes result: \n" + str(swapAxes) + "\n")

base = np.array([[[11,10], [2,3]], [[20,1], [6,7]]])

print("Showcasing the base structure in terms of usage for axis 2 swap: \n" + str(base) + "\n")

baseShape = base.shape

print("Showcasing the base shape allocation in terms of from the axis 2 allocation: \n" + str(baseShape) + "\n")

swapAxisTwo = np.swapaxes(base,0,2)

print("Showcase the swapped axes of axis 2: \n" + str(swapAxisTwo) + "\n")

#Before swap
###[[[11 10]
##  [ 2  3]]
##
## [[20  1]
##  [ 6  7]]]

#After swap
##[[[11 20]
##  [ 2  6]]
##
## [[10  1]
##  [ 3  7]]]

#Basically, after swap - the structure is:
# Before
# [[[a,b], [c,d]]  [[[11 10] [ 2 3]]
#  [[e,f] , [g,h]]   [[20 1] [ 6 7]]]
#   
# After
# [[[a,e], [c,g]] [[[11 20] [ 2 6]]  
#
#  [[b,f], [d,h]]    [[10 1] [3 7]]]
#

print("================= SHOWCASING OF numpy.swapaxes OVER ================\n")

#Index 6.2 - Slicing and np.index_exp

print("================= SHOWCASING OF the slicing dynamics of np and np.index_exp =============== \n")

#Now, to fully understand the documentation of numpy.take and the integration - we have to cover other areas first, as well.

#Showcasing some documentation of numpy.s_ to see how that works for slicing and indexing

base = np.array([1,2,3])

#      start index -v    v--Slice step
result = base[np.s_[0:  :5]]             #If start index is none, it's prsumed to be at the starting index, if end is none, it's the whole thing
#                        ^- end index  #if Slice step is too large, it is just discarded if it's taken out of bounds
# 
print("Showcasing the result from slicing of the np.s: \n" + str(result) + "\n")

#If we utilize np.index_exp, we get a sliced tuple object that can be used in construction of complex index expressions
result = np.index_exp[2::2]

print("Showcasing in terms of the sliced tuple object: \n" + str(result) + "\n")

print("================= SHOWCASING OF the slicing dynamics of np and np.index_exp  OVER =============== \n")

#Index 6.3 - numpy.take

print("======================== SHOWCASING OF numpy.take ======================\n")

#Take is utilized to access elements from indexes as follows

#Whilst there is a overall coverage on the speed optimisations in terms of circumventing certain partitions
#in terms of accessing with slice objects pre-done to eliminate a inner loop in nested partitions
#
#I feel like illustrating the basic dynamics of interactions, compared to speed optimisations, as of this document.

base = [4, 3, 5, 7, 6, 8] #The base to access from

print("Showcasing the base structure taken from: \n" + str(base) + "\n")
indexes = [0, 1, 4] #The indexes to access with

result = np.take(base, indexes) #The result of accessing the base with indexes

print("Showcasing the result from accessing with indexes of the base: \n" + str(result) + "\n")

indexes = [[0,1], [2,4]]
#The implicit typing in terms of the shape and dimensions is inherited in terms of indexes 
#Assuming that the dimensions of the indexing array is not 1D
result = np.take(base, indexes) #Formatting will be [[ ] [ ]] as per implicit hierarchy of shaping of dimensions

print("Showcasing of the result: \n" + str(result) + "\n") 

print("===================== SHOWCASING OF numpy.take OVER =======================\n")

#Index 6.4 - numpy.transpose

print("===================== SHOWCASING OF numpy.transpose ========================\n")

#We can, if we wish - perform transpositions in terms of reshapes as follows

base = np.arange(4).reshape((2,2))

print("Showcasing the base: \n" + str(base) + "\n")

#Inverse the dimensions in terms of the Structure
transpose = np.transpose(base)

print("Showcasing the transpose of the base structure: \n" + str(transpose) + "\n")

#Where of, we can transpose across specific indexes as well, if we wish
base = np.arange(30).reshape(5,3,2)

print("Showcasing base structure before transposing: \n" + str(base) + "\n")

transpose = np.transpose(base, (2,0,1)) #Basically flips it from Vertical to horizontal, along each column


print("Showcasing the transpose by virtue of specific indexes: \n" + str(transpose) + "\n")

print("========================= SHOWCASING OF TRANSPOSE OVER ======================\n")

#Index 6.5 - numpy.vsplit

print("========================= SHOWCASING OF numpy.vsplit ==========================\n")

#In terms of vsplit, the default is axis=0

base = np.arange(16.0).reshape(4,4) #Initialize the base structure

print("Showcasing the base structure: \n" + str(base) + "\n")

firstSplit = np.vsplit(base,2)

print("Showcasing in terms of running vsplit with index 2 as start: \n" + str(firstSplit) + "\n")

twoIndexSplit = np.vsplit(base, np.array([3, 6]))

print("Showcasing in terms of running vsplit with designation points of indexes: \n" + str(twoIndexSplit) + "\n")

#We can of course, run the split on higher dimension examples as well

base = np.arange(8.0).reshape(2, 2, 2) #Still runs split along axis 0, vertically

print("Showcasing base structure: \n" + str(base) + "\n")

                       #  v indexes or sections
base = np.vsplit(base, 2)

print("Showcasing base having been split with np.vsplit along axis 0: \n" + str(base) + "\n")

print("========================== SHOWCASING OF vsplit OVER ====================\n")

#Index 6.6 - numpy.vstack

print("========================== SHOWCASING OF vstack =========================\n")

#Stacks on the vertical

base1 = np.array([1,2,3])
base2 = np.arange(3)

print("Showcasing two basic structures to operate on: \n" + str(base1) + "\n\n" + str(base2) + "\n")

result = np.vstack((base1, base2))

print("Showcasing the result: \n" + str(result) + "\n")

#Where of if we wish to have separation for each level of index, we can do as follows
base1 = np.array([[1], [2], [3]])
base2 = np.array([[2], [3], [4]])

print("Showcasing the two basic structures: \n" + str(base1) + "\n\n" + str(base2) + "\n")



result = np.vstack((base1,base2))

print("Showcasing the result: \n" + str(result) + "\n")

print("================== END OF SHOWCASING OF vstack ================\n")

#Index 6.7 - numpy.astype

print("================== SHOWCASING OF astype ===================\n")

base = np.array([1, 2, 2.5])

print("Showcasing in terms of the base structure: \n" + str(base) + "\n")


base[0] = 20.3 #Designation before copy has been intialized

#Showcase the forcing of type conversion in terms of int casting with declaration to cause a copy of said typing
base2 = base.astype(int)

base[2] = 30.1 #Modify to illustrate that copy is not affected by assignment of original

print("Showcasing the base structure that was modified: \n" + str(base) + "\n\nShowcasing the casted version: " + str(base2) + "\n")

print("================== SHOWCASING OF astype OVER =================\n")

#Index 6.8 - numpy.atleast_1d

print("================== SHOWCASING OF numpy.atleast_1d =============\n")

#If we wish, we can convert inputs to shapes of at least 1 dimension, with these calls

#Convert the base structure to 1D
base = np.atleast_1d(1.0)

print("Showcasing of the base element having been converted to 1d Element: \n" + str(base) + "\n")

base = np.arange(24).reshape(3,2,4)

print("Showcasing the base element of the 24 range unto shape of 3,2,2: \n" + str(base) + "\n")

reformedBase = np.atleast_1d(base)

print("Showcasing the reformedBase: \n" + str(reformedBase) + "\n") 

result = np.atleast_1d(base) is base #Run operations in terms of equality to showcase that the two are identical, as copies are only made if nessecary

print("Showcasing the result of comparison: \n" + str(result) + "\n")

#To showcase how the integration in terms of conversion of singular elements occurs, we can showcase as follows

base = np.atleast_1d(1, [3, 4]) #Loose paired elements will be formatted into own encasing

print("Showcasing the base structure: \n" + str(base) + "\n")

print("=================== SHOWCASING OF np.atleast_1d OVER ======================\n")

#Index 6.9 - numpy.atleast_2d

print("=================== SHOWCASING OF numpy.atleast_2d ========================\n")

#The slight difference of 2d is that copies are avoided and views of two or more dims are returned.

base = np.atleast_2d(3.0, 8.5, (3.3, 33))

print("Showcasing of the 2D encasing of the base: \n" + str(base) + "\n")

base = np.arange(3.0)
base2 = np.atleast_2d(base)

print("Showcasing base: \n" + str(base) + "\n")
print("Showcasing base2, having run np.atleast_2d: \n" + str(base2) + "\n")

#Checking for equivalence in terms of view
result = base2 is np.atleast_2d(base2)

print("Showcasing of the result of comparison: \n" + str(result) + "\n")

#Again, showcasing interactions of encasing of loose elements

base = np.atleast_2d(1, [1,2], [[1,2]])

print("Showcasing in terms of encapsulated isolated elements: \n" + str(base) + "\n")

print("==================== END OF SHOWCASING OF np.atleast_2d ===============\n")

#Index 7.0 - numpy.atleast_3d

print("==================== SHOWCASING OF numpy.atleast_3d ==================\n")

#We can, if we wish - reformulate the shaping and handling to be 3D, as can be showcased

base = np.atleast_3d(3.0)

print("Showcasing the base transformed: \n" + str(base) + "\n") #yields 1,3,1 shape

base = np.arange(12.0).reshape(4,3) #We can of course, run to account for 4,3 to become 3D as well with np.atleast_3d

print("Showcasing the base shape of the base structure: \n" + str(base.shape) + "\n")

base = np.atleast_3d(base)

print("Showcasing the shape of the new reformation: \n" + str(base.shape) + "\n")

print("=================== END OF SHOWCASING OF numpy.atleast_3d ===============\n")

#Index 7.1 - numpy.mat

print("=================== SHOWCASING OF numpy.mat ========================\n")

base = np.array([[1,2], [3,4]]) #We can transform the typing to be as a Matris
#This does not create a copy, it changes how hte form is interpreted

print("Showcasing the base structures typing: \n" + str(type(base)) + "\n")

base = np.asmatrix(base)

print("Showcasing the changed typing of base: \n" + str(type(base)) + "\n")

print("================== SHOWCASING OF numpy.mat OVER ===================\n")

#Index 7.2 - numpy.arange

print("================== SHOWCASING OF numpy.arange ===================\n")

#We can use the arange command to spawn arrays with start index/end index and stepping difference in indexing
base = np.arange(3)

print("Showcase the basic structure of the base: \n" + str(base) + "\n")

base = np.arange(3.0)

print("Showcase the range for float implicit typing: \n" + str(base) + "\n")

base = np.arange(3,7)

print("Showcase the range between 3 and 7: \n" + str(base) + "\n")

base = np.arange(3,11,2) #Showcase stepping between 3 and 10 with 2 length step
#Do denote that if hte stepping would be outside of the range, it is simply discarded or ignored.

print("Showcase the range between 3 and 10, 2 step difference: \n" + str(base) + "\n")

print("============= SHOWCASING OF numpy.arange OVER ================\n")

#Index 7.3 - numpy.array

print("============= SHOWCASING OF numpy.array ======================\n")

#We can utilize array to spawn arrays. Do denote, dtype can only be used for upcasting.
#Downcasting is done by .astype(t)

#The base default parameter of copying is true, i.e, a copy is returned

#We can gate wether sub classes are passed through, through subok - however,
#if false, it is forced to be base class.

#If we wish, we can specify the memory layout in terms of Ordering ordeal
#
# order         no copy         copy=True
#
# 'K'             unchanged     F & C order preserved, otherwise most similar order
#
# 'A'             unchanged     F order if input is F and not C, otherwise C order
#
# 'C'             C order         C order, C style - iterative horizontal
#
# 'F'             F order          F order, Fortran style - iterative vertical

#If copy is false, but one has to be made - it's made as a pseudo call to copy=true (i.e, as copy=true, but with some exceptions)

#Where of, we can prepend 1's to account for the desired dimensions with ndmin

#Note, if the implied Order is A - and it's an array of neither C or F order - and a copy is forced by dtype enforcement
#Then the result is not nessecarily C.

#Intialize the base structure
base = np.array([1,2,3])

print("Showcase the base structure: \n" + str(base) + "\n")

#Showcasing the upcasting
base = np.array([1, 2, 3.0]) #Forces all elements to be float point numerals

print("Showcasing the upcasted version: \n" + str(base) + "\n")

#Showcasing initialization of more than one dimension
base = np.array([[1, 2], [3, 4]])

print("Showcasing the base in terms of multidimensional integration: \n" + str(base) + "\n")

#Reinforcing minimum amount of dimensions needed
base = np.array([1, 2, 3], ndmin=2)

print("Showcasing the base in 2 dimensions: \n" + str(base) + "\n")

#Where of, we can run with typing as declared - if we wish
base = np.array([1, 2, 3], dtype=complex)

print("Showcasing the integration of explicit typing declared before initialization: \n" + str(base) + "\n")

#Where of, if we wish - we can denote data types that consist of more than one element
base = np.array([(1,2), (3,4)], dtype=[('a', '<i4'), ('b', '<i4')]) #The designation here, is that a spans 1,3 and b spans 2,4

print("Showcasing the base structure: \n" + str(base) + "\n\nType: " + str(base.dtype) + "\n")

access = base['a']

access2 = base['b']

print("Showcasing in terms of accessing of types which spans several values: \n" + str(access) + "\n")

print("Showcasing in terms of accessing of types which spans several values: \n" + str(access2) + "\n")

#Where of, we can also create an array from sub-classing

base = np.array(np.mat('1 2; 3 4')) #No subclass enforcement of typing occurs

base2 = np.array(np.mat('1 2; 3 4'), subok=True) #Subclass enforcement of typing occurs

print("Showcasing in terms of the base typing of the base structure: \n" + str(type(base)) + "\n") 

print("Showcasing in terms of the base typing of the base structure: \n" + str(type(base2)) + "\n")

print("======================= SHOWCASING OF np.array dynamics OVER ================\n")

#Index 7.4 - numpy.copy

print("======================= SHOWCASING OF numpy.copy =====================\n")

#We can, if we wish - interact with copying objects and parts
#Do note, this function equates to that of running intiialization with copy set to true

base = np.array([1, 2, 3]) #Basic intialized structure

z = np.copy(base)

print("Showcasing base structure: \n" + str(base) + "\n")

print("Showcasing the copy of the base: \n" + str(base) + "\n")

base[0] = 15

z[0] = 20

print("Showcasing of the base element: \n" + str(base) + "\n")

print("Showcasing of the copy: \n" + str(z) + "\n")

result = base[0] is z[0]

print("Showcasing of the comparison between base[0] and z[0]: \n" + str(result) + "\n")

print("======================== SHOWCASING OF numpy.copy OVER ============\n")

#Index 7.5 - numpy.empty

print("======================== SHOWCASING OF numpy.empty ==============\n")

#We can, if we wish - create an array with the given shape and type, without initializing entries

base = np.empty([3,3])

print("Showcasing in terms of random initialization of the base structure: \n" + str(base) + "\n")

#We can define what level of type we wish to have as well
base = np.empty([2, 2], dtype=int)

print("Showcasing in terms of random initialization of specific typing: \n" + str(base) + "\n")

print("========================= SHOWCASING OF numpy.empty OVER ==============\n")

#Index 7.6 - numpy.empty_like

print("========================= SHOWCASING OF numpy.empty_like ================\n")

#We can, if we wish - utilize empty_like to get a new array with the same shape and type, as the
#array being called up - however, with randomized initialized values

base = ([1,2,3], [4,5,6])

print("Showcasing of the base structure: \n" + str(base) + "\n")

#Where of, if we wish to implement randominitialization in terms of integer, as the implicit typing is adhering there
intBase = np.empty_like(base) #initializes with random int values, in accordance to shape/length of the base element

print("Showcasing in terms of np.empty_like(): \n" + str(intBase) + "\n")

base = np.array([[5., 10., 15.], [20., 25., 60.]])

floatBase = np.empty_like(base)  #Same as before in terms of shape/length, however - the typing is implicit to follow

print("Showcasing in terms of float random initialization: \n" + str(floatBase) + "\n")

print("========================= SHOWCASING OF numpy.empty_like OVER ===============\n")

#Index 7.7 - numpy.eye

print("========================= SHOWCASING OF numpy.eye ==========================\n")

#Where of, if we wish to get a 2d array with ones on the diagonal and zeros elsewhere

base = np.eye(2, dtype=int) #Run with a specific int designation across a 2x2 structure with 1's on the diag, 0 elsewhere

print("Showcasing in terms of the base of a 2x2: \n" + str(base) + "\n")

base = np.eye(3, k=1) #Where of we can construct a 3x3 structure with a diagonal off-set of 1, meaning we off-set by one upwards.

print("Showcasing in terms of the base of a 3x3, k1: \n" + str(base) + "\n")

print("=========================== END OF SHOWCASING OF numpy.eye ====================\n")

#Index 7.8 - numpy.fromfile

print("=========================== SHOWCASING OF numpy.fromfile ========================\n")

#If we wish, can construct a array from data in a text or a binary file

#Whilst this can be used for in terms of reading/writing, we should not rely on this in combination with tofile
#to account for data storage - as the binary files are not platform independent.

#i.e, no byte-order data or data-type information is saved. Data can be stored in the platform independent
# .npy format using save and load, instead.

#First we have to start with constructing an ndarray:

types = np.dtype([('time', [('min', int), ('sec', int)]), ('temp', float)])

print("Showcasing the types: " + str(types) + "\n")

base = np.zeros((1,), dtype=types)

print("Showcasing the base structure: \n" + str(base) + "\n")

 #Assign a temporary name to a filename from underlying OS calls

#base.tofile(os.tmpnam()) #Run writing to a file in terms of the designated filename

#readBase = np.fromfile(filename, dtype=dt)
#Whilst technically we could do the above, i cannot perform the script action on my own comptuer
#due to not yielding administrative rights in running the script due to account permissions.
#
#However, it does not matter all that much - the code is to illustrate, not to save/load.

#print("Showcasing the read data from the base structure: \n" + str(readBase) + "\n")

testSave = 'BaseName'

#However, the recommended way of doing it, is:
np.save(testSave, base)

baseLoad = np.load(testSave + '.npy')

print("Showcasing the baseLoad contents: \n" + str(baseLoad) + "\n")

print("======================== END OF SHOWCASING OF numpy.fromfile ===============\n")

#Index 7.9 - numpy.fromfunction

print("======================== SHOWCASING OF numpy.fromfunction ==================\n")

#If we wish, we can construct a structure based from a function integration, as can be showcased

base = np.fromfunction(lambda i, j: i, (3,3) , dtype=int) #Basically, utilize a lambda function to iterate over a structure
#to create a structure of 3x3, which in our case increments one per row
#So, it's 
# 0 0 0
# 1 1 1
# 2 2 2

print("Showcasing in terms of the base structure: \n" + str(base) + "\n")

#we can utilize other operations in terms of operations, of course

base = np.fromfunction(lambda i, j: i + (j * 2), (3,3), dtype=int)
#This one, gets us to have assignment to each element ot be
# i + (j * 2)
#
# Which is
# 0 + (0 * 2), 0 + (1 * 2), 0 + (2 * 2)
# 1 + (0 * 2), 1 + (1 * 2), 1 + (2 * 2)
# 2 + (0 * 2), 2 + (1 * 2), 2 + (2 * 2)

print("Showcasing in terms of the base structure: \n" + str(base) + "\n")

print("=================== SHOWCASING OF numpy.fromfunction OVER ==================\n")

#Index 8.0 - numpy.identity

print("=================== SHOWCASING OF numpy.identity =======================\n")

#This simply returns the identity array, which is a square array with ones on the main diagonal

base = np.identity(9) #When we define the identity of 9, we mean the diagonal of a system of 9x9
#where of the dtyping of the variables defaults to float points

print("Showcasing the identity of a diagonal system of 9x9: \n" + str(base) + "\n") 

print("==================== SHOWCASING OF numpy.identity OVER ================\n")


#Index 8.1 - numpy.linspace

print("==================== SHOWCASING OF numpy.linspace =====================\n")

#Basically iteratively defines a linear space of where we subsection the range and the amount of steps to have been taken
base = np.linspace(2.0, 10.0, num=5) #Construes a 5 step array, with endstep of 3

print("Showcasing base range between 2-10 with endstep 10: \n" + str(base) + "\n")

base = np.linspace(2.0, 10.0, num=5, endpoint=False) #basically, exclude endpoint in terms of division step, so
#if you have 5 steps, that is to have occured before endpoint, in our case 10 

print("Showcasing the base range between 2-10 with endstep < 10 : \n" + str(base) + "\n")

#Where of, if we wish to denote the step between each, we can call it with retstep
base = np.linspace(2.0, 10.0, num=7, retstep=True) 

print("Showcasing the base range between 2-0 with endstep 10, accounting for showing step: \n" + str(base) + "\n")

#Where of, if we wish to plot them - we can do so as well
amountofpoints = 10
baseHorizontal = np.ones(amountofpoints)

base1 = np.linspace(0, 10, amountofpoints, endpoint=True)
base2 = np.linspace(0, 10, amountofpoints, endpoint=False)

#plt.plot(base1, baseHorizontal, 'o') #The shape can take different designations 
#plt.plot(base2, baseHorizontal + 10, 'X') #Range to plot them in, and their respective form

#plt.ylim([-0.25, 15]) #This is the limit on the vertical, so go from -0.25 to 15
#plt.xlim([-2, 11]) #This is the limit on the horizontal, so go from 0 to 20
#Denote that scaling of hte plotting is directly proportional to how large the axises are

#plt.show() #Show the plot

print("======================= SHOWCASING OF numpy.linspace OVER ==================\n")

#Index 8.2 - numpy.mgrid

print("======================= SHOWCASING OF numpy.mgrid =========================\n")

base = np.mgrid[-1:7.5,-2:2] #In case of float point numbers, rounding down is accounted for

print("Showcasing of the base mesh grid: \n" + str(base) + "\n")

#Where of, we can initialize a mesh grid on the virtue of complex arguments with integrals integrated as well
#However, the real part is the amount of points to exist in division in terms of the range accounted for

base = np.mgrid[1:5:5j, 1:10:5j] #Where of the real part of the j, denotes the amount of steps
#DO note, the first arg runs on vertical, descending - thus, 1:5:5j the first part - runs on vertical descending
#The second arg runs on horizontal from start point to end point - thus, 1:10:5j the second part - runs on horizontal

print("Showcasing the second base mesh grid, where of complex real part denotes steps: \n" + str(base) + "\n")

base = np.mgrid[5:0:1j, 10:5:5j] #Do note, that if the step is 1 - basically no reduction in terms of the structure
#occurs. It basically stops at the initial base value

print("Showcasing in terms of the mesh grid, where we are using a startpoint that is > end point : \n" + str(base) + "\n")

print("==================== SHOWCASING OF numpy.mgrid OVER ========================\n")

#Index 8.3 - numpy.ogrid

print("==================== SHOWCASING OF numpy.ogrid ============================\n")

#Now, the difference between mgrid and ogrid - is that mgrid is a fleshed out mesh net structure.
#ogrid is a open grid mesh structure, which means - that we end up with a situation of where
#indexes are not filled out.

base = ogrid[-10:10:5j] #The real part of the complex argument, gets accounted for as being 5 stopping points, inclusively speaking

print("Showcasing the base structure with complex arguments: \n" + str(base) + "\n")

base = ogrid[10:11,0:10] #Runs first pattern on Vertical, second on Horizontal
#Do note, that you cannot iterative on a descending numeral pattern, as in 10:5, as that will give an empty array.
#Equally so, if you try to go 0 to -5

print("Showcasing the base structure in terms of without complex args: \n" + str(base) + "\n")

print("===================== SHOWCASING OF numpy.ogrid OVER ======================\n")

#Index 8.4 - numpy.ones

print("===================== SHOWCASING OF numpy.ones ============================\n")

#If we wish, we can construct a structure of which is the given shape and type - filled with 1's

base = np.ones(10) #Just a basic structure to showcase interplays

print("Showcasing the basic structure: \n" + str(base) + "\n")

base = np.ones((10), dtype=int) 
# Each respective argument is a modification to the structure, as in - 10 is just 10 elements,
# had we written 2 there after - it'd be 2 on the width, so 20 elements, 2 rows of 10 elements
#
# Where of the each argument added adds to the structure in terms of dimensions etc.

print("Showcasing structure of specific typing in base structure: \n" + str(base) + "\n" + str(base.dtype) + "\n\nShape: " + str(base.shape) + "\n")

base = np.ones(5, dtype=float)

print("Showcasing the float integrated structure: \n" + str(base) + "\n")

print("Showcasing without actual type encapsulation: \n" + str(base.dtype) + "\n\nShape: " + str(base.shape) + "\n")

base = np.ones((10,2, 3)) 

print("Showcasing the base structure with np.ones((10,2)) designation: \n" + str(base) + "\n")

print("Showcasing the base shape of a designated structure of np.ones((10,2)): \n" + str(base.shape) + "\n")

print("======================= SHOWCASING OF np.ones OVER =======================\n")

#Index 8.5 - numpy.ones_like

print("======================= SHOWCASING OF numpy.ones_like ======================\n")

#Where of if we wish to yield an array that supports the same shape and type as the given array
#We can utilize the numpy.ones_like

#Create a base range in terms of elements
base = np.arange(24)

print("Showcasing of the base structure: \n" + str(base) + "\n")

#Perform reshaping of the base element
baseReshape = base.reshape(4,3,2) 

print("Showcasing of the reshaped base structure: \n" + str(baseReshape) + "\n")

onesBaseShape = np.ones_like(baseReshape)

print("Showcasing of the copied structure but with 1's: \n" + str(onesBaseShape) + "\n")

#Do keep in mind, that typing is kept as well - as in, if a Float range is copied - then that structure is kept.

print("========================== SHOWCASING OF numpy.ones_like OVER ===================\n")

#Index 8.6 - numpy.zeros

print("========================== SHOWCASING OF numpy.zeros =======================\n")

#Where of the basic structure integration and interaction is similar to numpy.ones - we can showcase it
#in terms of different integrations

base = np.zeros(5) #Do note that it defaults to Float typing in initialization

print("Showcasing the base structure with float default parameter declaration: \n" + str(base) + "\n")

base = np.zeros(5, dtype=int) #Denotes integer dtype initialization instead

print("Showcasing the base structure with int declaration: \n" + str(base) + "\n")

#We can of course, operate on the dimensions of the structure as well
base = np.zeros((2,1))
print("Showcasing of the base structure of 0's, shape 2,1: \n" + str(base) + "\n\nShape: " + str(base.shape) + "\n")

#We can keep adding dimensions and interacting with the structure to modify the composition as we see fit
base = np.zeros((2,2))
print("Showcasing of the base structure of 0's, shape 2,2: \n" + str(base) + "\n\nShape: " + str(base.shape) + "\n")

base = np.zeros((3,), dtype=[('x', 'i4'), ('y', 'i4'), ('a', 'i4'), ('b', 'i4'), ('c', 'i4')]) #Custom designate a dtype 
#Utilizes broadcating in terms of amount of typings - where the np.zeros parameter defines the amount
#of sections in terms of array parts, where of the dtyping represent amount of elements per part.
#
#For instance, as above, 5 typings - delegates 5 elements per part, where of we have 3 parts due to initialization of parameters.

print("Showcasing the base structure in terms of initialized with several dtypes: \n" + str(base) + "\n")

print("====================== SHOWCASING OF numpy.zeros =========================\n")

#Index 8.7 - numpy.zeros_like

print("====================== SHOWCASING OF numpy.zeros_like ======================\n")

#In a sense, numpy.zeros_like is akin to numpy.ones_like 

base = np.arange(30)

base = base.reshape((5,3,2)) # 5 * 3 * 2 = 15 * 2 = 30
#i.e, 5 parts - 2 width, 3 parts on each part

print("Showcasing the base structure in terms of 5,3,2 shape: \n" + str(base) + "\n")

baseZeros = np.zeros_like(base)

print("Showcasing in terms of similar structure and elements of base designated with zeros: \n" + str(baseZeros) + "\n")

#Of course, float typing designation is the same - as it's carried over.

print("========================== SHOWCASING OF numpy.zeros_like OVER ===============\n")


#This constituted part 1 of my Series of learning Numpy.
#These documents are meant to showcase and illustrate how i go along and learn to develop in Numpy/Integrate into
#the Science parts of Python development.
