import numpy as np
import math
import random




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
        tmp2 = np.e-np.exp(1./self.variable_num*np.sum(np.cos(variables*2.*np.pi)))
        return tmp1+tmp2


# Function to be Minimized (Rosenbrocks Valley). Solution ->  f(x) = 0; xi = 1
class RosenBrocksValley:
    min_values=[-5,-5]
    max_values=[5,5]
    
    def fitness_function(self,variables_values = [0,0]):
        func_value = 0
        last_x = variables_values[0]
        for i in range(1, len(variables_values)):
                func_value = func_value + (100 * math.pow((variables_values[i] - math.pow(last_x, 2)), 2)) + math.pow(1 - last_x, 2)
        return func_value


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


#Minima ->  f(x, y) = -959.6407; x=512, y=404.2319
class EggHolder:
    min_values=[-512,-512]
    max_values=[512,512]

    def fitness_function(self,values):
        x=values[0]
        y=values[1]
        return -(y+47)*np.sin(np.sqrt(np.abs(y+(x/2)+47)))-x*np.sin(np.sqrt(np.abs(x-(y+47))))



#Minima ->  f(x1, x2) = -1.0316; x1 = 0.0898, x2 = -0.7126 or x1 = -0.0898, x2 = 0.7126
class SixHumpCamelBack:
    min_values=[-5,-5]
    max_values=[5,5]
    def fitness_function(self,variables_values):
        func_value = 4*variables_values[0]**2 - 2.1*variables_values[0]**4 + (1/3)*variables_values[0]**6 + variables_values[0]*variables_values[1] - 4*variables_values[1]**2 + 4*variables_values[1]**4
        return func_value









