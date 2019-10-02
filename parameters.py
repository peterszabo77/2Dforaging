WITHGUI=True
DRAWING=True

#FORAGING MODEL
L=100 #side length of arena
FORAGER_SIZE = 1 # forager size (radius)
FORAGER_INTERACTION_RADIUS = 10.0 # forager interaction radius
FORAGER_SPEED = 10.0 # forager speed
FORAGER_DENSITY = 0.001 # density of foragers
N_FORAGERS = int(L*L*FORAGER_DENSITY)
FOOD_BR = 0.002 # per unit area
FOOD_DR = 0.1 # per food item
FOOD_SIZE = 1 # forager size (radius)
ELASTIC_COLLISIONS = False

DT=0.05 # integration step
DIR_RESOL = 10 # resolution/number of movement directions
N_ACTIONS = DIR_RESOL
FORAGER_VISRESOL = 11 #resolution of perceptive field (odd number)

#REINFORCEMENT LEARNING PARAMETERS
BATCH_SIZE = 100
TRAINING_STEPS = 10000 # number of training steps
TRAINING_START = 1000 # start training after some iterations
TRAINING_INTERVAL = 100 # run a training step every ... game iterations
MEMORY_SIZE = TRAINING_INTERVAL*N_FORAGERS # replay memory size
COPY_STEPS = 1 # copy the critic to the actor every ... training steps
DISCOUNT_RATE = 0.95
EPSILON = 0.01
INIT_LEARNING_RATE = 0.001 #critic DQN's learning rate
LEARNING_RATE_LIFETIME = 500 # learning steps between halving the learning rate
EXPLORATION_STRATEGY =  'e-greedy' # 'e-greedy' or 'softmax'
