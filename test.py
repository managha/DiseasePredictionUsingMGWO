import numpy as np
import math
import random
import benchMarks

Max_iter=10000
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

def gwoLoop(benchMarkObject):

    min_values=benchMarkObject.min_values
    max_values=benchMarkObject.max_values
    fitness_function=benchMarkObject.fitness_function
    
    alpha    = alpha_position(min_values, max_values,target_function = fitness_function)
    print("initial alpha score:",fitness_function(alpha))
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
                alpha = Positions[i][:]
                print("iteration ",l,": new alpha found:",alpha_score)
            if (fitness > alpha_score and fitness < beta_score):
                beta_score = fitness
                beta = Positions[i][:]
            if (fitness > alpha_score and fitness > beta_score and fitness < delta_score):
                delta_score = fitness
                delta = Positions[i][:]


        
        a = 2 - 2 * ((l / Max_iter)**2) # a decreases linearly fron 2 to 0
        
        # Update the Position of search agents including omegas
        for i in range(0, SearchAgents_no):
            for j in range(0, dimension):

                r1 = random.random()  # r1 is a random number in [0,1]
                r2 = random.random()  # r2 is a random number in [0,1]

                A1 = 2 * a * r1 - a
                if A1<0:
                    A1=2*2*r1-2
                C1 = 2 * r2
                
                D_alpha = abs(C1 * alpha[j] - Positions[i][j])
                X1 = alpha[j] - A1 * D_alpha    

                r1 = random.random()
                r2 = random.random()

                A2 = 2 * a * r1 - a
                if A2<0:
                    A2=2*2*r1-2
                C2 = 2 * r2
        
                D_beta = abs(C2 * beta[j] - Positions[i][j])
                X2 = beta[j] - A2 * D_beta         

                r1 = random.random()
                r2 = random.random()

                A3 = 2 * a * r1 - a
                if A3<0:
                    A3=2*2*r1-2
                C3 = 2 * r2
                
                D_delta = abs(C3 * delta[j] - Positions[i][j])
                X3 = delta[j] - A3 * D_delta
                
                x=(X1 + X2 + X3)/3
                        
                Positions[i][j]=x
    print("final optimized value: ",alpha_score)




gwoLoop(benchMarks.Ackley())

# print(benchMarks.Rastrigin().fitness_function([0,0]))