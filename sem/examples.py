# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:49:47 2016

@author: FrankKaMa
"""

import sem

priors={}
priors['A']=0.2
priors['C']=0.3
priors['G']=0.2
priors['T']=0.3

# transition matrix for nucleotides
row_1=[0.9996,0.0001,0.0002,0.0001] 
row_2=[0.0001,0.9996,0.0001,0.0002]
row_3=[0.0002,0.0001,0.9996,0.0001]
row_4=[0.0001,0.0002,0.0001,0.9996]


eps_bs=5
eps_edge_length=0.0000000001   
sem_steps=3
rho=0.3
M=20

# Create an instance of a forest object
F=sem.Forest()

# Iniciate it with 3 trees with 8 nodes each 
F.rand_forest(3,8)

# Return tree with name tree_1 
tree1=F.get_tree('tree_1')

# Set variables of states and transition probabilities
print sem.set_states(['A','G','C','T'])
print sem.set_transition_probabilities([row_1,row_2,row_3,row_4])


# Simulate a multiple alignment
D=sem.simulate_multiples_alignment(tree1,M,priors)

# Choose a tree as inital tree
tree_guess=F.get_tree("tree_2")
    
# Run Structural EM Algorithm
estimated_tree=sem.structural_em(eps_edge_length,priors,D,rho,sem_steps,tree_guess)

# Show results  
estimated_tree.print_edges()
    