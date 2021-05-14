import numpy as np
import math
import random

#------------------------------Unimodel------------------------------------------------

# Minimum value : f(x1,...,xn)=f(0,...,0)=0
class Sphere:
    max_values = np.array([1000] * 2)  # nearly inf
    min_values = np.array([-1000] * 2)  # nearly inf

    def fitness_function(self, variables):
        return np.sum(np.square(variables))


# Function to be Minimized (Rosenbrocks Valley). Solution ->  f(x) = 0; xi = 1
class RosenBrocksValley:
    min_values = [-5, -5]
    max_values = [5, 5]

    def fitness_function(self, variables_values=[0, 0]):
        func_value = 0
        last_x = variables_values[0]
        for i in range(1, len(variables_values)):
            func_value = func_value + (100 * math.pow((variables_values[i] - math.pow(last_x, 2)), 2)) + math.pow(
                1 - last_x, 2)
        return func_value

#global_optimum_solution = 0.

class SumOfDifferentPower:

    max_values = [1,1]
    min_values = [-1,-1]
    variable_num=2

    def fitness_function(self, variables):
        tmp = 0
        for i in range(self.variable_num):
            tmp += np.power(np.absolute(variables[i]),i+2)
        return tmp

# Minimum value: f(0,0)=0
class Matyas:

    max_values = np.array([10.]*2)
    min_values = np.array([-10.]*2)

    def fitness_function(self, variables):
        tmp1 = 0.26*(np.power(variables[0],2)+np.power(variables[1],2))
        tmp2 = 0.48*variables[0]*variables[1]
        return tmp1-tmp2

#global_optimum_solution = 0
class Step:
    max_values = np.array([100] * 2)
    min_values = np.array([-100] * 2)

    def fitness_function(self, variables):

        return np.sum(np.floor((variables + [0.5])) ** 2)

#global_optimum_solution = 0
class Quartic:
    max_values = np.array([1.28] * 2)
    min_values = np.array([-1.28] * 2)

    def fitness_function(self, variables):
        w = [i for i in range(len(variables))]
        np.add(w, 1)
        s = np.sum(np.multiply(w, np.power(variables, 4)))
        return s



#--------------------------------------------------Multimodel-----------------------------------

# Minimum Value: f(0,0)=0
class Rastrigin:
    min_values=[-5.12,-5.12]
    max_values=[5.12,5.12]
    variable_num=2

    def fitness_function(self, variables):
        tmp1 = 10 * self.variable_num
        tmp2 = 0
        for i in range(self.variable_num):
            tmp2 += np.power(variables[i],2)-10*np.cos(2*np.pi*variables[i])
        return tmp1+tmp2

# Minimum Value: f(0,0)=0
class Ackley:
    max_values = [32.768,32.768]
    min_values = [-32.768,-32.768]
    variable_num=2
    
    
    def fitness_function(self, variables):
        tmp1 = 20.-20.*np.exp(-0.2*np.sqrt(1./self.variable_num*np.sum(np.square(variables))))
        tmp2 = np.e-np.exp(1./self.variable_num*np.sum(np.cos(list(map(lambda x:x*2.*np.pi,variables)))))
        return tmp1+tmp2


#Minima ->  f(x, y) = -959.6407; x=512, y=404.2319
class EggHolder:
    min_values=[-512,-512]
    max_values=[512,512]

    def fitness_function(self,values):
        x=values[0]
        y=values[1]
        return -(y+47)*np.sin(np.sqrt(np.abs(y+(x/2)+47)))-x*np.sin(np.sqrt(np.abs(x-(y+47))))

#global_optimum_solution = 0
class Griewank:

    max_values = np.array([600.]*2)
    min_values = np.array([-600.]*2)
    variable_num=2



    def fitness_function(self, variables):
        tmp1 = 0
        tmp2 = 1
        for i in range(self.variable_num):
            tmp1 += np.power(variables[i],2)
            tmp2 = tmp2*np.cos(variables[i]/np.sqrt(i+1))
        return tmp1/4000-tmp2+1


# Minimum value: f(0,0)=0
class SchafferN2:

    max_values = np.array([100.]*2)
    min_values = np.array([-100]*2)

    def fitness_function(self, variables):
        tmp1 = np.power(np.sin(np.power(variables[0],2)-np.power(variables[1],2)),2)-0.5
        tmp2 = np.power(1+0.001*(np.power(variables[0],2)+np.power(variables[1],2)),2)
        return 0.5+tmp1/tmp2

#global_optimum_solution = -1.8013
class Michalewicz:

    max_values = np.array([np.pi]*2)
    min_values = np.array([0.]*2)
    variable_num=2

    def fitness_function(self, variables):
        m = 10
        tmp1 = 0
        for i in range(self.variable_num):
            tmp1 += np.sin(variables[i])*np.power(np.sin((i+1)*np.power(variables[i],2)/np.pi),2*m)
        return -tmp1

#global_optimum_solution = 0
class XinSheYang:

    max_values = np.array([2.*np.pi]*2)
    min_values = np.array([-2.*np.pi]*2)
    variable_num=2


    def fitness_function(self, variables):
        tmp1 = 0
        tmp2 = 0
        for i in range(self.variable_num):
            tmp1 += np.absolute(variables[i])
            tmp2 += np.sin(np.power(variables[i],2))
        return tmp1*np.exp(-tmp2)

#--------------------------------------------------------------Others-----------------------------------------------

# Minimum Value: f(3,0.5)=0
class Beale:
    min_values=[-4.5,-4.5]
    max_values=[4.5,4.5]
    def fitness_function(self,x):
	    return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2


#Minimum value: f(0,-1)=3
class GoldsteinPrice:

    min_values=[-2,-2]
    max_values=[2,2]

    def fitness_function(self, variables):
        tmp1 = (1+np.power(variables[0]+variables[1]+1,2)*(19-14*variables[0]+3*np.power(variables[0],2)-14*variables[1]+6*variables[0]*variables[1]+3*np.power(variables[1],2)))
        tmp2 = (30+(np.power(2*variables[0]-3*variables[1],2)*(18-32*variables[0]+12*np.power(variables[0],2)+48*variables[1]-36*variables[0]*variables[1]+27*np.power(variables[1],2))))
        return tmp1*tmp2

#Minimum value: f(1,3)=0
class Booth:

    max_values = np.array([10.]*2)
    min_values = np.array([-10.]*2)

    def fitness_function(self, variables):
        tmp1 = np.power(variables[0]+2*variables[1]-7,2)
        tmp2 = np.power(2*variables[0]+variables[1]-5,2)
        return tmp1+tmp2

#Minimum value: f(-10,1)=0
class BukinN6:

    max_values = np.array([-5.,3.])
    min_values = np.array([-15.,-3.])

    def fitness_function(self, variables):
        tmp1 = 100*np.sqrt(np.absolute(variables[1]-0.01*np.power(variables[1],2)))
        tmp2 = 0.01*np.absolute(variables[0]+10)
        return tmp1+tmp2




# Minimum value: f(1,1)=0
class LeviN13:

    max_values = np.array([10.]*2)
    min_values = np.array([-10.]*2)


    def fitness_function(self, variables):
        tmp1 = np.power(np.sin(3*np.pi*variables[0]),2)
        tmp2 = np.power(variables[0]-1,2)*(1+np.power(np.sin(3*np.pi*variables[1]),2))
        tmp3 = np.power(variables[1]-1,2)*(1+np.power(np.sin(2*np.pi*variables[1]),2))
        return tmp1+tmp2+tmp3

# Minimum value: f(0,0)=0
class ThreeHumpCamel:

    max_values = np.array([5.]*2)
    min_values = np.array([-5.]*2)

    def fitness_function(self, variables):
        return 2*np.power(variables[0],2)-1.05*np.power(variables[0],4)+np.power(variables[0],6)/6+variables[0]*variables[1]+np.power(variables[1],2)

# Minimum value: f(pi,pi)=-1 pi=3.14
class Easom:

    max_values = np.array([100.]*2)
    min_values = np.array([-100.]*2)

    def fitness_function(self, variables):
        return -1.0*np.cos(variables[0])*np.cos(variables[1])*np.exp(-(np.power(variables[0]-np.pi,2)+np.power(variables[1]-np.pi,2)))



#Minima ->  f(x1, x2) = -1.0316; x1 = 0.0898, x2 = -0.7126 or x1 = -0.0898, x2 = 0.7126
class SixHumpCamelBack:
    min_values=[-5,-5]
    max_values=[5,5]
    def fitness_function(self,variables_values):
        func_value = 4*variables_values[0]**2 - 2.1*variables_values[0]**4 + (1/3)*variables_values[0]**6 + variables_values[0]*variables_values[1] - 4*variables_values[1]**2 + 4*variables_values[1]**4
        return func_value


# Minimum value: f(-0.54719,-1.54719)=-1.9133
class McCormick:

    max_values = np.array([4.]*2)
    min_values = np.array([-1.5,-3.])

    def fitness_function(self, variables):
        tmp1 = np.sin(variables[0]+variables[1])+np.power(variables[0]-variables[1],2)
        tmp2 = -1.5*variables[0]+2.5*variables[1]+1
        return tmp1+tmp2


# Minimum value: f(0.,1.25313)=0.292579
class SchafferN4:

    max_values = np.array([100.]*2)
    min_values = np.array([-100]*2)

    def fitness_function(self, variables):
        tmp1 = np.power(np.cos(np.sin(np.absolute(np.power(variables[0],2)-np.power(variables[1],2)))),2)-0.5
        tmp2 = np.power(1+0.001*(np.power(variables[0],2)+np.power(variables[1],2)),2)
        return 0.5+tmp1/tmp2








