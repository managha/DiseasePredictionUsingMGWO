import numpy as np
import math
import random

Max_iter=700
SearchAgents_no=15
dimension=2

def initial_position(min_values, max_values, target_function):
    position=[]
    for i in range(SearchAgents_no):
        pos=[0]*len(min_values)
        position.append(pos)

    for i in range(0, SearchAgents_no):
        for j in range(0, len(min_values)):
             position[i][j] = random.uniform(min_values[j], max_values[j])
    return position

def alpha_position(min_values, max_values,target_function):
    alpha = [0]*dimension
    for j in range(0, dimension):
        alpha[j] = random.uniform(min_values[j], max_values[j])
    return alpha

# Function: Initialize Beta
def beta_position(min_values, max_values,target_function):
    beta = [0]*dimension
    for j in range(0, dimension):
        beta[j] = random.uniform(min_values[j], max_values[j])
    return beta

# Function: Initialize Delta
def delta_position(min_values, max_values,target_function):
    delta =  [0]*dimension
    for j in range(0, dimension):
        delta[j] = random.uniform(min_values[j], max_values[j])
    return delta

def gwoLoop(min_values, max_values,fitness_function):
    
    alpha    = alpha_position(min_values, max_values,target_function = fitness_function)
    beta     = beta_position(min_values, max_values,target_function = fitness_function)
    delta    = delta_position(min_values, max_values,target_function = fitness_function)
    Positions = initial_position(min_values, max_values,target_function = fitness_function)

    for l in range(0, Max_iter):
        for i in range(0, SearchAgents_no):

            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(dimension):
                Positions[i][j] = np.clip(Positions[i][j], min_values[j], max_values[j])
            
            # Calculate objective function for each search agent
            fitness = fitness_function(Positions[i])
            alpha_score=fitness_function(alpha)
            beta_score=fitness_function(beta)
            delta_score=fitness_function(delta)
            
            
            if fitness < alpha_score :
                alpha_score = fitness
                alpha = Positions[i]
            if (fitness > alpha_score and fitness < beta_score):
                beta_score = fitness
                beta = Positions[i]
            if (fitness > alpha_score and fitness > beta_score and fitness < delta_score):
                delta_score = fitness
                delta = Positions[i]
        
        a = 2 - l * ((2) / Max_iter) # a decreases linearly fron 2 to 0
        
        # Update the Position of search agents including omegas
        print(alpha_score)
        for i in range(0, SearchAgents_no):
            for j in range(0, dimension):

                r1 = random.random()  # r1 is a random number in [0,1]
                r2 = random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                
                D_alpha = abs(C1 * alpha[j] - Positions[i][j])
                X1 = alpha[j] - A1 * D_alpha    

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a
                C2 = 2 * r2
        
                D_beta = abs(C2 * beta[j] - Positions[i][j])
                X2 = beta[j] - A2 * D_beta         

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                
                D_delta = abs(C3 * delta[j] - Positions[i][j])
                X3 = delta[j] - A3 * D_delta
                
                x=(X1 + X2 + X3)/3
                        
                Positions[i][j]=x
    return alpha_score


#Minima ->  f(x1, x2) = -1.0316; x1 = 0.0898, x2 = -0.7126 or x1 = -0.0898, x2 = 0.7126
def six_hump_camel_back(variables_values):
    func_value = 4*variables_values[0]**2 - 2.1*variables_values[0]**4 + (1/3)*variables_values[0]**6 + variables_values[0]*variables_values[1] - 4*variables_values[1]**2 + 4*variables_values[1]**4
    return func_value

six_hump_min=[-5,-5]
six_hum_max=[5,5]

#Minima ->  f(x, y) = -959.6407; x=512, y=404.2319
def eggHolder(values):
    x=values[0]
    y=values[1]
    return -(y+47)*np.sin(np.sqrt(np.abs(y+(x/2)+47)))-x*np.sin(np.sqrt(np.abs(x-(y+47))))

eggHolder_min=[-512,-512]
eggHolder_max=[512,512]


# Function to be Minimized (Rosenbrocks Valley). Solution ->  f(x) = 0; xi = 1
def rosenbrocks_valley(variables_values = [0,0]):
    func_value = 0
    last_x = variables_values[0]
    for i in range(1, len(variables_values)):
        func_value = func_value + (100 * math.pow((variables_values[i] - math.pow(last_x, 2)), 2)) + math.pow(1 - last_x, 2)
    return func_value

def beale_fn(x):
	return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2

beale_min=[-4.5,-4.5]
beale_max=[4.5,4.5]


print(gwoLoop(eggHolder_min,eggHolder_max,eggHolder))

