# -*- coding: utf-8 -*-

import numpy as np 
import random
import time
import copy

'''
Graph and Tree Objects
'''

class Node:
    def __init__(self, node):
        self.id = node
        self.adjoined = {}

    def __str__(self):
        return str(self.id) + ' adjoined to: ' + str([x.id for x in self.adjoined])
    
    def neighbors(self):
        return [x.id for x in self.adjoined]    
     
    def add_neighbor(self, neighbor, edge_length=0):
        self.adjoined[neighbor] = edge_length

    def remove_neighbor(self, neighbor):
        del self.adjoined[neighbor]
    
    def get_connections(self):
        return self.adjoined.keys()  

    def get_id(self):
        return self.id

    def get_edge_length(self, neighbor):
        return self.adjoined[neighbor]

class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0

    def __iter__(self):
        return iter(self.vert_dict.values())
        
    def set_graph(self,vertices_dict,number_vertices):
        self.vert_dict={a:b for a,b in vertices_dict.items()}
        self.num_vertices=number_vertices
        
    def get_vert_dict(self):
        return self.vert_dict
        
    def get_num_vertices(self):
        return self.num_vertices
        
    def get_node(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            raise ValueError

    def get_time(self,node1,node2):    
        v1=self.get_node(node1)
        v2=self.get_node(node2)
        try: 
            return v1.get_edge_length(v2)
        except KeyError:
            print("No edge between node %s and node %s!")% (node1,node2)
        
    def add_node(self, node):
        self.num_vertices = self.num_vertices + 1
        new_node = Node(node)
        self.vert_dict[node] = new_node
        return new_node

    def remove_node(self ,node):
        for neighbor in self.neighbors(node):
            self.remove_edge(node,neighbor)
        del self.vert_dict[node]
        
    def add_edge(self, frm, to, cost = 0):
        if frm not in self.vert_dict:
            self.add_node(frm)
        if to not in self.vert_dict:
            self.add_node(to)
        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def remove_edge(self,node1,node2):
        if((node1 in self.vert_dict) and (node2 in self.vert_dict)):
            self.vert_dict[node1].remove_neighbor(self.vert_dict[node2])
            self.vert_dict[node2].remove_neighbor(self.vert_dict[node1])
    
    def nodes(self):
        return self.vert_dict.keys()
    
    def edges(self):
        edge_list=[]
        for v in self:
            for w in v.get_connections():
                vid = v.get_id()
                wid = w.get_id()
                if(((vid,wid) not in edge_list) and ((wid,vid) not in edge_list) ):
                    if(wid<=vid):
                        edge_list.append((wid,vid))
                    else:
                        edge_list.append((vid,wid))
        return sorted(edge_list)

    def print_edges(self):
        edge_list=[]
        for v in self:
            for w in v.get_connections():
                vid = v.get_id()
                wid = w.get_id()
                if(((vid,wid) not in edge_list) and ((wid,vid) not in edge_list)):
                    if(wid<=vid):
                        print '( %s , %s, %3d)'  % ( wid, vid, v.get_edge_length(w))
                        edge_list.append((wid,vid))
                    else:
                        print '( %s , %s, %3d)'  % ( vid, wid, v.get_edge_length(w))
                        edge_list.append((vid,wid))

    def has_node(self, node):
        if (node in self.vert_dict.keys()):
            return True
        else:
            return False
            
    def has_edge(self, node1, node2):
        if (((node1,node2) in self.edges()) or ((node2,node1) in self.edges())):
            return True
        else:
            return False
            
    def neighbors(self,node):
        ver=self.get_node(node)
        ver_dic=self.vert_dict[ver.get_id()]
        return ver_dic.neighbors()

    def degree(self,node):
        return len(self.neighbors(node))
        
    def bfs_nodes(self, root):
        list_of_nodes=self.nodes()
        list_of_nodes.remove(root)
        bfs_list=[root]
        while(list_of_nodes!=[]):
            for visited_node in bfs_list:
                for node in list(self.neighbors(visited_node)):
                    if(node in list_of_nodes):
                        list_of_nodes.remove(node)
                        bfs_list.append(node)
        return bfs_list
        
    def dfs_nodes(self, root, path):
        path.append(root)
        for node in sorted(self.neighbors(root),reverse=True):
            if not node in path:
                path=self.dfs_nodes(node, path)
        return path

    def dfs_edges(self,root):
        n=self.dfs_nodes(root,[])
        dfs_edges=[(n[0],n[1])]
        visited_nodes=[n[0],n[1]]
        l=1
        k=2
        while(len(n)!=len(visited_nodes)):
            if(self.has_edge(visited_nodes[-1],n[k]) is True):
                dfs_edges.append((visited_nodes[-1],n[k]))
                visited_nodes.append(n[k])
                k=k+1
            else: 
                for l in range(len(visited_nodes)-1,-1,-1):
                    if(self.has_edge(visited_nodes[-l],n[k]) is True):
                        if(((visited_nodes[-l],n[k]) not in dfs_edges) and ((n[k],visited_nodes[-l]) not in dfs_edges)):
                            dfs_edges.append((visited_nodes[-l],n[k]))        
                            visited_nodes.append(n[k])
                            k=k+1
                            break
        return dfs_edges

    def rand_tree(self,nodes):
        self.vert_dict = {}
        self.num_vertices = 0
        forks=(nodes-2)/2
        leafs=forks+2
        if((nodes<4) or (forks%1!=0)):
            raise ValueError("It is not possible to create a bifurcating tree, because the wrong number of nodes is passed!")
        t1=1
        t2=5
        list_of_leafs=list(np.arange(1,leafs+1))
        if(forks==1):
            for i in range(1,leafs+1):
                 n=len(list_of_leafs)
                 choose_leaf=random.randint(0, n-1)
                 self.add_edge(str(list_of_leafs[choose_leaf]),str(4), random.randint(t1, t2))
                 list_of_leafs.remove(list_of_leafs[choose_leaf])
        else:
            for i in range(leafs+1,forks+leafs+1):
                if(i==leafs+1):
                    #Add first leaf
                    n=len(list_of_leafs)
                    choose_leaf=random.randint(0, n-1)
                    self.add_edge(str(list_of_leafs[choose_leaf]),str(i), random.randint(t1, t2))
                    list_of_leafs.remove(list_of_leafs[choose_leaf])
                    
                    # Add second leaf
                    n=len(list_of_leafs)
                    choose_leaf=random.randint(0, n-1)
                    self.add_edge(str(list_of_leafs[choose_leaf]),str(i), random.randint(t1, t2))
                    list_of_leafs.remove(list_of_leafs[choose_leaf])
                    
                    #Add edge to internal node
                    k=i+1
                    self.add_edge(str(i),str(k), random.randint(t1, t2))
                
                elif(i==forks+leafs):
                     #Add first leaf
                    n=len(list_of_leafs)
                    choose_leaf=random.randint(0, n-1)
                    self.add_edge(str(list_of_leafs[choose_leaf]),str(i), random.randint(t1, t2))
                    list_of_leafs.remove(list_of_leafs[choose_leaf])
                    
                    # Add second leaf
                    self.add_edge(str(list_of_leafs[0]),str(i), random.randint(t1, t2))
                    list_of_leafs.remove(list_of_leafs[0])
                else:
                    #Add leaf
                    n=len(list_of_leafs)
                    choose_leaf=random.randint(0, n-1)
                    self.add_edge(str(list_of_leafs[choose_leaf]),str(i), random.randint(t1, t2))
                    list_of_leafs.remove(list_of_leafs[choose_leaf])
                    # Add edge to internal node
                    k=i+1
                    self.add_edge(str(i),str(k), random.randint(t1, t2))
        
class Forest(Graph):
    def __init__(self):
        self.tree_dict = {}
        self.num_trees = 0
    
    def add_tree(self, name):
        self.num_trees = self.num_trees + 1
        new_tree = Graph()
        self.tree_dict[name] = new_tree
    
    def remove_tree(self, name):
        self.num_tree=self.num_trees-1
        del self.tree_dict[name] 
        
    def trees(self):
        return sorted(self.tree_dict.keys())
    
    def get_tree(self,name):
        return self.tree_dict.get(name)
    
    def update_tree(self,name,new_tree):
        self.remove_tree(name)
        self.num_trees = self.num_trees + 1
        self.tree_dict[name] = new_tree
        
    def rand_forest(self,trees,nodes):
        for k in range(1,trees+1):
            name="tree_"+str(k)
            self.add_tree(name)
            tree_k=self.get_tree(name)
            tree_k.rand_tree(nodes)

'''
    Functions 
'''
def set_states(letters):
    '''
    Set the states of the markov chain

    letters: A list of letters representing the differnet states    
    '''    
    if(type(letters) is not list):
        raise ValueError (" The input data letters is not a list!")
    global states   
    states={}
    letters.sort()
    for l in range(0,len(letters)):
        states[letters[l]]=l
    return states

def set_transition_probabilities(rows):
    '''
    Set the matrix of transition probabilities
    
    rows:  A list with rows of the transition matrix in alphabetical order
    '''
    if(len(rows)!=len(states)):
        raise ValueError ("The number of rows doesn't match the number of states!")
    elif(type(rows) is not list):
        raise ValueError (" The input data rows has to be a list!")
    for row_index in range(0,len(rows)):
        if(len(rows[row_index])!=len(states)):
            raise ValueError ("The row length does not match the number of states!")
       
    # transition matrix for nucleotides
    global transition_matrix
    transition_matrix=np.matrix(rows)
    return transition_matrix


def find_parent(T, root, node):
    '''
    Returns the neighbor node closest to the root node
    
    T:      graph
    root:   node designated to be the root 
    node:   node in Graph T
    '''
    preorder_nodes=T.dfs_nodes(root,[])  
    neighbors=T.neighbors(node)    
    if(node==root):
        raise ValueError("The given node is equal to the root, can't find a parent node!")
    elif(len(neighbors)==1):
       return neighbors[0]
    elif(len(neighbors)>1):
        for i in range(len(preorder_nodes)):
                if(preorder_nodes[i] in neighbors):
                    return preorder_nodes[i]
    else:
        raise ValueError("Node %c has no parent node" % node)

def find_children(G, root, node):
    '''
    Returns a list of neighbors to a node, which are successors.  
    
    G:      graph
    root    node designated to be the root    
    node    node of graph G
    '''
    neighbors=G.neighbors(node)
    if(node==root):
        return neighbors
    else:
        parent=find_parent(G, root, node)
        if parent in neighbors:
            neighbors.remove(parent)
        return neighbors
    
def pruning(G,leafs):
    sorted_nodes=sorted(list(G.nodes()), key=int) 
    nodes=sorted_nodes[leafs:]
    for node in nodes:
        if (G.degree(node)==1):
            G.remove_node(node)
            pruned_node=node
            break
        elif(G.degree(node)==2):
            neighbors=list(G.neighbors(node))
            m=neighbors[0]
            o=neighbors[1]
            new_time=G.get_time(m,node)+G.get_time(node,o)   
            G.remove_node(node)
            G.add_edge(m, o,new_time)
            pruned_node=node
            break
        else:
            pruned_node=None
    return pruned_node
    
def indroduce_new_node(G,node,new_root,pruned_node, eps):
    # Find parent node and remove edge to parent node 
    parent=find_parent(G, new_root, node)
    edge_length=G.get_time(node,parent)  
    G.remove_edge(node,parent)
    # indroduce new node 
    new_node=pruned_node
    # add edges to new node 
    G.add_edge(new_node, node, eps)
    G.add_edge(new_node, parent, edge_length)
    # add child of node n to new node 
    # take first child 
    children=find_children(G, new_root, node)
    edge_length=G.get_time(node,children[0])    #
    G.remove_edge(node, children[0])
    G.add_edge(children[0], new_node, edge_length)
    
def create_bifurcating_tree(eps,G,root):
    '''
    Input:
    eps:    edge length to new indroduced node 
    G:      graph
    root:   a node designated to be the root
    
    Output:
    bifurcating tree
    '''
    leafs=(len(G.nodes())/2)+1
    if(leafs==4):
        forks=['5','6']
        forks.remove(root)
        if(G.degree(root)<G.degree(forks[0])):
            root=forks[0]
    while(True):
        pruned_node=pruning(G,leafs)
        if(pruned_node==None):
            break
        elif(pruned_node==root):
            list_of_nodes=sorted(list(G.nodes()), key=int)
            #list_of_nodes.remove(root)
            count_nodes=0
            for node in list_of_nodes[leafs:]:
                count_nodes=count_nodes+1
                if(G.degree(node)>3):
                    if(count_nodes==len(list_of_nodes[leafs:])):
                        new_root=node                    
                        G.add_edge(root, node, eps)
                        # Add first leaf to new indroduced node
                        edge_length_1=G.get_time(node,list_of_nodes[0])      
                        G.remove_edge(node,list_of_nodes[0])
                        G.add_edge(root,list_of_nodes[0],edge_length_1)
                        # Add second leaf to new indroduced node 
                        edge_length_1=G.get_time(node,list_of_nodes[0])    
                        G.remove_edge(node,list_of_nodes[1])
                        G.add_edge(root,list_of_nodes[1], edge_length_1)
                        break
                elif(G.degree(node)<=3):
                    new_root=node
                    break
        elif(G.degree(root)<=3):
            new_root=root
        elif(G.degree(root)>3):
            list_of_nodes=sorted(list(G.nodes()), key=int)
            count_nodes=0
            for node in list_of_nodes[leafs:]:
                count_nodes=count_nodes+1
                if(G.degree(node)>3):
                    if(count_nodes==len(list_of_nodes[leafs:])):
                        new_root=root                    
                        G.add_edge(root, pruned_node, eps)
                        # Add first leaf to new indroduced node
                        edge_length_1=G.get_time(root,list_of_nodes[0])        
                        G.remove_edge(root,list_of_nodes[0])
                        G.add_edge(pruned_node,list_of_nodes[0], edge_length_1)
                        # Add second leaf to new indroduced node 
                        edge_length_1=G.get_time(root,list_of_nodes[1])                    
                        G.remove_edge(root,list_of_nodes[1])
                        G.add_edge(pruned_node,list_of_nodes[1], edge_length_1)
                        break
                elif(G.degree(node)<=3):
                    new_root=node
                    break
        nodes=G.nodes() 
        for node in nodes:
            if((int(node)<leafs+1)and(G.degree(node)>1)):
                indroduce_new_node(G,node,new_root,pruned_node,eps)
                break
            elif((int(node)>leafs)and(G.degree(node)>3)):
                indroduce_new_node(G,node,new_root,pruned_node,eps)
                break
        
def create_subset(seq,n,N,m):
    '''
    Function creates a subset of a given sequence data. 
       
    seq:    dictionaray with sequences 
    N:      number of sequnces 
    m:      lengh of sequences 
    '''
    if(len(seq)<N):
        raise ValueError("Set of sequence data is to small")
    subset={}
    species=sorted(seq, key=int)
    count=1
    for specie in species:
        if(int(specie)<n):
            continue
        elif(int(specie)>N):
            return subset
        elif(len(seq[specie])<m):
            print specie
            raise ValueError("Sequence of specie is to short!")
        else:
            subset[str(count)]=seq.get(specie)[:m]
            count=count+1
    return subset

def create_subtree(node, root, T):
    '''
    
    node    node of T, from which on we prune the subtree 
    root    the root of T
    T:      graph
    '''
    if(node==root):
        return T
    elif(T.has_node(root) is False):
        raise ValueError('Root is not in tree T!')
    elif(T.has_node(node) is False):
        raise ValueError("Node is not int tree T!")
    elif(node_type(node, T)=="leaf"):
        subtree=Graph()
        subtree.add_node(node)
        return subtree
    else:
        subtree=Graph()
        T1=copy.deepcopy(T)
        if(node in T.neighbors(root)):
            parent=root
        else:
            parent=find_parent(T, root, node)
        T1.remove_node(parent)
        edges=T1.dfs_edges(node)
        for k in range(len(edges)): 
            subtree.add_edge(edges[k][0],edges[k][1],T.get_time(edges[k][0],edges[k][1]))
        return subtree    

def dynamic_programming_matrix(internal_nodes):
    '''
    Creates a matrix, with a column for each internal node and the number of states to the power of intenal node rows.
    
    internal_nodes: an integer, corresponding to the number of internal nodes  
    '''
    n=len(states)
    dim=n**internal_nodes
    row=np.repeat(0.0, internal_nodes)
    matrix_list=[]
    for i in range(dim):
        matrix_list.append(row)
    return np.matrix(matrix_list)

def node_type(node, G):
    '''
    Returns the type of a node, where we distinguish between 5 types of nodes. 
    '''
    if(G.has_node(node) is False):
        raise ValueError('Graph G has no node %c!' % node)
    neighbors=G.neighbors(node)
    if(G.degree(node)==1):
        return "leaf"
    elif(len(G.nodes())==4):
        # Type0 correspond to an internal node with no internal nodes neighbors 
        return "type0"
    elif((G.degree(neighbors[0])>1) and (G.degree(neighbors[1])>1) and (G.degree(neighbors[2])>1)):
        # Type 3 correpond to an internal node with 3 internal nodes neighbors
        return "type3"
    elif(((G.degree(neighbors[0])==1 and G.degree(neighbors[1])==1)) or ((G.degree(neighbors[0])==1 and G.degree(neighbors[2])==1)) or ((G.degree(neighbors[1])==1 and G.degree(neighbors[2])==1))):
        # Type 1 correspond to an internal node with 1 internal nodes neighbors
        return "type1"
    else:
        # Type 2 correspond to an internal node with 2 interanl node neighbor
        return "type2"

def neighbor_leafs(node,T):
    '''
    Returns all leafs, which are neighbors to node 
    
    node:             node of T
    T:                graph  
    '''
    neighbor_leafs=T.neighbors(node)
    neighbors=T.neighbors(node)
    for node in neighbors:
        if(T.degree(node)>1):
            neighbor_leafs.remove(node)
    return neighbor_leafs

def neighbor_forks(node,T):
    '''
    Returns all forks, which are neighbors to node 
    
    node:             node of T
    T:                graph  
    '''
    neighbor_forks=T.neighbors(node)
    neighbors=T.neighbors(node)
    for node in neighbors:
        if(T.degree(node)<3):
            neighbor_forks.remove(node)
    return neighbor_forks[0]

def breadth_first_search(root,G):
    '''
    Returns list of nodes of graph G in the breadth first search order. 
    
    root:  initial node to start breadth first search 
    G:     graph 
    '''
    list_of_nodes=G.nodes()
    list_of_nodes.remove(root)
    bfs_list=[]
    bfs_list.append(root)
    while(list_of_nodes!=[]):
        for visited_node in bfs_list:
            for node in G.neighbors(visited_node):
                if(node in list_of_nodes):
                    list_of_nodes.remove(node)
                    bfs_list.append(node)
    return bfs_list

def transition_probability(a,b,t):
    '''
    Computes the transition probability from state a to state b over the time horizon t
    
    a:                  a state 
    b:                  a state
    t:                  a positiv real number 
    '''
    if((a not in states) or (b not in states)):
        raise ValueError('Parameter %s or %s is not a state!' % (a,b))
    elif((t is None) or (t<1)):
        print t
        raise ValueError('Parameter type of variable t is not allowed!')
    elif(t==0):
        if(a==b):
            return 1
        elif(a!=b):
            return 0
    elif(t%1==0):
        new_matrix=transition_matrix**(t)
        probability=new_matrix[states[a],states[b]]
        if(probability>1):
            print t
            print a
            print b
            raise ValueError
        return probability
    else:
        t_int=int(t)
        #t_frac=int((t-t_int)*10)
        new_matrix=transition_matrix**(t_int)
        return new_matrix[states[a],states[b]] 

def upward_message(i,j,m,seq, a, T):
    '''
    Computes the upward message u_i->j.
    
    i:              a node of tree T
    j:              a node of tree T
    m:              the position in the sequence 
    seq:            a dicitonary of sequences 
    a:              a nucleotide for which the upward-message is calculated 
    T:              a bifurcating Tree 
    '''
    if(T.has_edge(i,j) is False):
         raise ValueError('Tree has no edge between node %s and node %s!' % (i,j))
    elif(len(a)!=1):
        raise ValueError("The input parameter a is not length one!" )
    
    # Create subtree with all nodes evolved from node i 
    new_tree=create_subtree(i,j,T)
        
    # Add new node j to subtree
    new_tree.add_edge(i,j,T.get_time(i,j)) 

    # Create list of leafs and internal nodes. By following the tree from node j in breadth first search order
    leaf_list=[]
    fork_list=[]
    bfs_nodes=new_tree.bfs_nodes(j)   
    for node in bfs_nodes:
        if(new_tree.degree(node)==3):
            fork_list.append(node)
        elif(new_tree.degree(node)==1):
            leaf_list.append(node)
        else:
            raise ValueError('Tree is not bifurcating!')

    # Create new sequence dicitonary  
    new_seq={}
    new_seq[j]=a
    for node in leaf_list:
        if(node==j):
            continue 
        else:
            new_seq[node]=seq.get(node)
                              
    if(len(fork_list)==0):
        return transition_probability(new_seq[i][m],new_seq[j],new_tree.get_time(i,j)) 
    else:
        # Count number of internal nodes
        internal_nodes=len(fork_list)
    
        # Create matrix to store results
        result_matrix=dynamic_programming_matrix(internal_nodes)
        
        # Visiting the first internal node
        visited_nodes=0
        node=fork_list[0]
        k=0
        # The first visited internal node is type1
        if(node_type(node, new_tree)=="type1"): 
            t1=new_tree.get_time(leaf_list[0],node) 
            n1=new_seq[leaf_list[0]]
            t2=new_tree.get_time(leaf_list[1],node) 
            n2=new_seq[leaf_list[1]][m]                            
            for x in states:
                result_matrix[k,visited_nodes]=transition_probability(n1,x,t1)*transition_probability(x ,n2,t2) 
                k=k+1  
            visited_nodes=1
        # The first visited internal node is type2
        elif(node_type(node, new_tree)=="type2"):
            leafs=neighbor_leafs(node,new_tree)
            t1=new_tree.get_time(leafs[0],node) 
            n1=new_seq[leafs[0]] 
            for x in states:
                result_matrix[k,visited_nodes]=transition_probability(n1,x,t1) 
                k=k+1 
            visited_nodes=1
        # The first visited internal node is type0
        elif(node_type(node, new_tree)=="type0"):
             leafs=neighbor_leafs(node,new_tree)
             leafs.remove(str(j))
             t1=new_tree.get_time(str(j),node)
             t2=new_tree.get_time(leafs[0],node) 
             t3=new_tree.get_time(leafs[1],node) 
             n1=new_seq[str(j)]  
             n2=new_seq[leafs[0]][m]
             n3=new_seq[leafs[1]][m]
             for x in states:
                 result_matrix[k,0]=transition_probability(n1,x,t1)*transition_probability(x ,n2,t2)*transition_probability(x ,n3,t3) 
                 k=k+1
             result=result_matrix[0:,0]
             return sum(result)               
        # Visiting the other internal nodes 
        for node in fork_list[1:]: 
            fork_type=node_type(node, new_tree)
            k=0
            dim=len(states)**(visited_nodes)
            
            # The internal node has two neighbors which are internal nodes
            if(fork_type=="type2"):
                leafs=neighbor_leafs(node,new_tree)
                forks=find_parent(new_tree, j, node) 
                t1=new_tree.get_time(forks,node) 
                t2=new_tree.get_time(leafs[0],node) 
                n1=new_seq[leafs[0]][m]
                for l in range(dim):
                    y=sorted(states)[l%len(states)]
                    for x in states:
                        result_matrix[k,visited_nodes]=transition_probability(x,y,t1)*transition_probability(y,n1,t2)*result_matrix[l ,visited_nodes-1]
                        k=k+1
                visited_nodes=visited_nodes+1
            # The internal node has three neighbors which are internal nodes
            elif(fork_type=="type3"):
                forks=find_parent(new_tree, j, node) 
                t1=new_tree.get_time(forks,node)
                for l in range(dim):
                    y=sorted(states)[l%len(states)]
                    for x in states:
                        result_matrix[k,visited_nodes]=transition_probability(y,x,t1)*result_matrix[l,visited_nodes-1]
                        k=k+1        
                visited_nodes=visited_nodes+1
            # The internal node has one neighbor which is an inernal node
            elif(fork_type=="type1"):
                leafs=neighbor_leafs(node,new_tree)
                forks=neighbor_forks(node,new_tree)
                t1=new_tree.get_time(leafs[0],node) 
                t2=new_tree.get_time(leafs[1],node) 
                t3=new_tree.get_time(forks[0],node) 
                n1=new_seq[leafs[0]][m]
                n2=new_seq[leafs[1]][m] 
                for l in range(dim):
                    y=sorted(states)[l%len(states)] 
                    for x in states:
                        result_matrix[k,visited_nodes]=transition_probability(n1,x,t1)*transition_probability(x ,n2,t2)*transition_probability(x ,y,t3)*result_matrix[l,visited_nodes-1]
                        k=k+1   
                visited_nodes=visited_nodes+1
    result=result_matrix[0:,-1]
    return sum(result)

def Upward_message(i,j,m,seq, a, T):
    '''
    Computes the upward message U_i->j.
    
    i:              a node of tree T
    j:              a node of tree T
    m:              the position in the sequence 
    seq:            a dicitonary of sequences 
    a:   a sequence for which the upward-message is calculated 
    T:              a bifurcating Tree 
    '''
    if(T.has_edge(i,j) is False):
        raise ValueError('Tree has no edge between node %s and node %s!' % (i,j))
    elif(len(a)!=1):
        raise ValueError("The input parameter %s has is too long!" % a)
    
    # If i is a leaf, return the deterministic result 
    if(T.degree(i)==1):
        if(seq[i][m]==a):
            return 1
        else:
            return 0 
    
    # If i is an internal node
    neighbors=T.neighbors(i)
    neighbors.remove(j)
    u1=upward_message(neighbors[0],i,m,seq, a, T)
    u2=upward_message(neighbors[1],i,m,seq, a, T)
    result=u1*u2
    return result
    
def marginal_probability(i,j,m,seq,T,priors):
    '''
    Compute the probability P(x_1,...x_N | T,t)
    
    i:           a node of tree T
    j:           a node of tree T
    m:           the position in the sequence 
    seq:         a dicitonary of sequences 
    T:           a bifurcating tree  
    priors:      a dictionary object with keys equal to the states 
    '''
    probability=0
    for x in states:
        probability=probability+priors[x]*Upward_message(i,j,m,seq, x, T)*upward_message(j,i,m,seq, x, T)
    return probability 

def conditional_probability_1(i,j,m,a,seq,T,priors):
    '''
    Calculates the conditonal probability P(X_i=a | x_1,...,x_N,T,t) 
    
    i:          a node of tree T
    j:          a node of tree T
    m:          a position in the sequence 
    a:          an element of states
    seq:        a dicitonary of sequences
    T:          a bifurcating Tree
    priors:     a dictionary object with keys equal to the states
    '''
    if(a not in states):
         raise ValueError('You have to specify %s to be a state!' % a)   
    else:
        mp=0
        for x in states:
            U=Upward_message(i,j,m,seq, x, T)
            u=upward_message(j,i,m,seq, x, T)
            mp=mp+priors[x]*U*u
            if(x==a):
                U_message=U
                u_message=u
        if(mp==0):
            raise ZeroDivisionError
        probability=(priors[a]*U_message*u_message)/mp
        return probability 
    
def conditional_probability_2(i,j,m,a,b,seq,T,priors):
    '''
    Calculates the conditonal probability P(X_i=a, X_j=b | x_1, ..., x_N,T,t) 
    
    i:          a node of tree T
    j:          a node of tree T
    m:          a position in the sequence 
    a:          an element of states
    b:          an element of states or None
    seq:        a dicitonary of sequences
    T:          a bifurcating Tree
    priors:     a dictionary object with keys equal to the states
    '''
    mp=0
    for x in states:
        U=Upward_message(i,j,m,seq, x, T)
        mp=mp+priors[x]*U*upward_message(j,i,m,seq, x, T)
        if(x==a):
            U_message=U
    numerator=priors[a]*U_message*transition_probability(a,b,T.get_time(i,j))*Upward_message(j,i,m,seq, b, T)
    if(mp==0):
        raise ZeroDivisionError
    return numerator/mp

def approximated_expected_counts(root,seq,T,priors): 
    '''
    1.  Create a dicitonary for all condional probabilities of one node  
    2.  Create a dicitonary with expected_counts of all edges in T
    3.  Add the approximated expected counts for all other links to the dictionary 
    
    Returns a dictionary with keys (i, j, a, b) and the approximated expected counts as values.  
    '''
    fork_list=[]
    nodes=T.nodes()
    nodes.sort()
    for node in nodes:
        if(T.degree(node)==3):
            fork_list.append(node)
    
    edge_list=T.dfs_edges(root) 
    edge_list.sort()   
    expected_counts={} 
    M=len(seq[list(seq)[0]])
    
    #t0=time.clock()
    # Create a dicitonary for the conditional probabilities 
    dic_conditional_probabilities={}
    for node in fork_list:
        if(node!=root):
            for x in states:
                for m in range(M):
                    key=(node,m,x)
                    parent=find_parent(T,root, node)
                    dic_conditional_probabilities[key]=conditional_probability_1(node,parent,m,x,seq,T,priors) 
        else:
            for x in states:
                for m in range(M):
                    key=(node,m,x)
                    neighbor=T.neighbors(node)[0]
                    dic_conditional_probabilities[key]=conditional_probability_1(node,neighbor,m,x,seq,T,priors)
                    
    #t1=time.clock()-t0
    #print "Time Conditional Probabilities 1: %3.2f" % t1     
    # Compute expected counts for edges 
    for edge in edge_list:
        # For i is a leaf and j is a fork 
        if((T.degree(edge[0])==3) and (T.degree(edge[1])==1)):
            for m in range(M):
                n1=seq[edge[1]][m]
                for x in states:
                    key=(edge[1],edge[0],n1,x)
                    key_prop=(edge[0],m,x)    
                    if(key in expected_counts):
                        expected_counts[key]=expected_counts.get(key)+dic_conditional_probabilities.get(key_prop)
                    else:
                        expected_counts[key]=dic_conditional_probabilities.get(key_prop)
                                                          
        # For i and j are forks
        elif((T.degree(edge[0])==3) and (T.degree(edge[1])==3)):
            for m in range(M):
                for y in states:
                    for x in states:
                        if(edge[1]<edge[0]):
                            key=(edge[1],edge[0],x,y)
                        else:
                            key=(edge[0],edge[1],y,x)
                        if(key in expected_counts):
                            expected_counts[key]=expected_counts.get(key)+conditional_probability_2(edge[1],edge[0],m,x,y,seq,T,priors)
                        else:
                            expected_counts[key]=conditional_probability_2(edge[1],edge[0],m,x,y,seq,T,priors)
        else:
            raise ValueError('Edge tupel %s has nodes sorted the wrong way around!' % edge)
    #t2=time.clock()-t0-t1
    #print "Time Conditional Probabilities 2: %3.2f" % t2
    # Compute expected counts for link, which are no edges      
    for index1 in range(len(nodes)):
        for index2 in range(index1+1,len(nodes)):
            if(T.has_edge(nodes[index1],nodes[index2])):
                continue
            # print (nodes[index1],nodes[index2])
            # i and j are leafs 
            if(T.degree(nodes[index1])==1 and T.degree(nodes[index2])==1):
                for m in range(M):
                    n1=seq[nodes[index1]][m]
                    n2=seq[nodes[index2]][m]
                    key=(nodes[index1],nodes[index2],n1,n2)
                    expected_counts[key]=1
            # if i is a leaf and j is a fork
            elif(T.degree(nodes[index1])==1 and T.degree(nodes[index2])==3):
                for m in range(M):
                    n1=seq[nodes[index1]][m]
                    for x in states:
                        key=(nodes[index1],nodes[index2],n1,x)
                        key_prop=(nodes[index2],m,x)
                        if(key in expected_counts):    
                            expected_counts[key]=expected_counts.get(key)+dic_conditional_probabilities.get(key_prop)   
                        else:
                            expected_counts[key]=dic_conditional_probabilities.get(key_prop)  
            # if i is a fork and j is a leaf
            elif(T.degree(nodes[index1])==3 and T.degree(nodes[index2])==1):
                for m in range(M):
                    n2=seq[nodes[index2]][m]
                    for x in states:
                        key=(nodes[index1],nodes[index2],y,n2)
                        key_prop=(nodes[index1],m,x)
                        if(key in expected_counts):    
                            expected_counts[key]=expected_counts.get(key)+dic_conditional_probabilities.get(key_prop) 
                        else:
                            expected_counts[key]=dic_conditional_probabilities.get(key_prop)  
            # if i and j are forks
            else:
                for m in range(M):
                    for y in states:
                        for x in states:
                            key=(nodes[index1],nodes[index2],x,y)
                            key_prop_x=(nodes[index1],m,x)
                            key_prop_y=(nodes[index2],m,y)    
                            if(key in expected_counts):
                                expected_counts[key]=expected_counts.get(key)+(dic_conditional_probabilities.get(key_prop_x)*dic_conditional_probabilities.get(key_prop_y)) 
                            else:
                                expected_counts[key]=dic_conditional_probabilities.get(key_prop_x)*dic_conditional_probabilities.get(key_prop_y) 
    return expected_counts    

'''
    Structural EM algorithm 
'''

def L_local(expected_counts,i,j,priors,t):
    '''
    Returns the sum of S_ij(a,b)*[log(p_(a->b)(t))-log(p_b)] for all a and b in states. 
    '''
    L=0
    for a in states:
        for b in states:
            key=(i,j,a,b)
            if(key in expected_counts):
                S_ij=expected_counts.get(key)
                if(t==0 and a!=b):
                    raise ValueError("L_local can't be computed for an edge length of 0!")
                tp=np.log(transition_probability(a,b,t))
                pr=np.log(priors[b])
                L=L+S_ij*(tp-pr)
    if(type(L) is float or long):
        return L
    else:
        print L
        raise ValueError

def optimize_link_length(eps,expected_counts,i,j,max_step,priors):
        '''
        Optimization stops eighter after max_steps or if the absolute improvement of L_local(...) from t_k to t_k+1 is less than eps
        
        Input:
        eps:                the mimimum improvment from one step to the next 
        expected_counts:    dictionary with keys (i,j) and values weight_ij
        i:                  node
        j:                  node
        max_step:           maximum steps for optimization
        priors:             dicitonary of prior probabilities
                
        Output:
        optimized link length between node i and node j
        '''
        #initial guess
        t_0=1
        t_1=2
        step_size=1
        path=[1]
        old_guess=L_local(expected_counts,i,j,priors,t_0)
        new_guess=L_local(expected_counts,i,j,priors,t_1)
        step=0
        moving_up=1
        while(abs(new_guess-old_guess)>eps):
            path.append(t_1)
            if(new_guess>old_guess):
                if(moving_up==1):
                    step_size=2*step_size
                else:
                    if(step_size/2>1):
                        step_size=step_size/2
                    else:
                        step_size=1
                old_guess=new_guess
                t_0=t_1
                t_1=t_1+step_size     
                new_guess=L_local(expected_counts,i,j,priors,t_1)
                moving_up=1
            else:
                if(moving_up==0):
                    step_size=2*step_size
                else:
                    if(step_size/2>1):
                        step_size=step_size/2
                    else:
                        step_size=1
                old_guess=new_guess
                t_0=t_1
                t_1=t_1-step_size 
                new_guess=L_local(expected_counts,i,j,priors,t_1)
                moving_up=0
            step=step+1
            #print (step,t_1,new_guess)
            if(t_1 in path):
                if(old_guess>new_guess):
                    t_1=t_0
                    new_guess=old_guess
                break
            elif(t_1==1):
                new_guess=L_local(expected_counts,i,j,priors,t_1)
                break
            elif(step==max_step):
                break
        return (t_1,new_guess)

def optimize_links(eps, expected_counts,priors,rho,max_steps,T):
    '''
    Returns a dicitonary with keys (i,j) and values (t_optimized, weight_ij)
    
    Input:
    eps                 absolute minimal improvment from on step to the other  
    expected_counts     dictionary with keys (i,j) and values weight_ij
    priors              dicitonary of prior probabilities 
    steps               maximum number of steps for each optimization 
    T                   bifurcating tree
    '''
    link_lengths={}
    nodes=T.nodes()
    nodes.sort()
    weights=[]
    for index1 in range(len(nodes)):
        for index2 in range(index1+1,len(nodes)):
            links=optimize_link_length(eps,expected_counts,nodes[index1],nodes[index2],max_steps,priors)
            weight=links[1]
            if(type(weight) is np.matrix):
                weight=weight[0,0]
            weights.append(weight)
            link_lengths[(nodes[index1],nodes[index2])]=(links[0],weight) 
    # estimate standard derivation for simulated annealing 
    sigma=np.std(weights)*rho
    for index1 in range(len(nodes)):
        for index2 in range(index1+1,len(nodes)):
            # perturbed edge weights with noise 
            noise=np.random.normal(0, sigma,1)[0]
            tw=link_lengths[(nodes[index1],nodes[index2])]
            link_lengths[(nodes[index1],nodes[index2])]=(tw[0],tw[1]+noise)
    
    return link_lengths.copy()

def likelihood_complete_data(seq,root,T,priors):
    '''
    Computes the likelihood in the complete data scenario
    
    Input:
    seq:    dictionary with sequence data with keys as string of the node number
    root:   node in T designated a as root 
    T:      bifurcating tree
    priors: dictionary of prior probabilities with keys of states
    
    Functions used: transition_probability(a,b,t)
    '''
    if(T.has_node(root) is False):
        raise ValueError('Wrong root node passed!')           
    edge_list=list(T.bfs_edges(root))
    likelihood=1
    M=len(seq[root])    
    for m in range(M):
        nucleotide=seq[root][m]
        likelihood=likelihood*priors[nucleotide]
        for edge in edge_list:
            t=T[edge[0]][edge[1]].get('time')
            a=seq[edge[0]][m]
            b=seq[edge[1]][m]
            likelihood=likelihood*transition_probability(a,b,t)
            #print likelihood
    return likelihood
    
def likelihood_incomplete_data(i,j,priors, D, T):   
    '''
    Calculates the likelihood P(D=d|T,t)
    '''
    M=len(D[list(D)[0]])
    likelihood=1
    for m in range(M):
        likelihood=likelihood*marginal_probability(i,j,m,D,T,priors)
    return likelihood 
    
def maximum_spanning_tree(optimized_links, number_nodes):
    '''
    Returns the maximum spanning tree
    
    Input: 
    optimized_links  dicitonary wtih key (i, j) and value (t_optimized, weight_ij)
    number_nodes     amount of nodes in the graph  
    '''
    max_tree=Graph() 
    nodes=[]
    # sort optimized_links for decreasing weights w_ij
    links=sorted(optimized_links.items(), key=lambda x: x[1][1],reverse=True)
    nodes.append(links[0][0][0])
    nodes.append(links[0][0][1])
    max_tree.add_edge(links[0][0][0],links[0][0][1],links[0][1][0])
    
    # Adding edges to the tree where one node is in the tree and the other not.
    # Searching for this edge in the list sorted by weight 
    for k in range(number_nodes-1):
        for link in links[1:]:
            if(((link[0][0] in nodes) and (link[0][1] not in nodes)) or ((link[0][0] not in nodes) and (link[0][1] in nodes))):
                max_tree.add_edge(link[0][0],link[0][1],link[1][0])
                nodes.append(link[0][0])
                nodes.append(link[0][1])        
                break
    return max_tree

def structural_em(eps,priors,D,rho,max_steps,T):
    '''
    A structural em alogrithm for phylogenetic inference 
    
    Input:
    eps:            parameter for edge length optimization 
    priors:         dictionary of prior probabilities with keys of states
    D:              multiples sequence alignment in format of a dicitonary 
    rho:            annealing parameter
    max_steps:      maximum number of steps, which the em algorithm runs
    T:              inital guess for a tree
        
    Output:
    bifurcating tree  
    '''
    number_of_nodes=len(T.nodes())
    last_node=number_of_nodes-1
    root=str(last_node)
    steps=0
    best_tree=None
    new_rho=rho
    
    neighbors=neighbor_forks(root,T)
    likelihood_best=likelihood_incomplete_data(neighbors[0],root,priors, D, T)
    print "Likelihood(100 digits): %.100f" % likelihood_best
    
    while(steps<max_steps):
        
        steps=steps+1
        t0=time.clock()
        # E-Step
        counts=approximated_expected_counts(root,D,T,priors)
        t1=time.clock()-t0
        print "Step %d: E-Step,    Time: %3.2f" % (steps,t1) 
                
        # M-Step I 
        max_steps_edge_length_opt=5
        links=optimize_links(eps,counts,priors,new_rho,max_steps_edge_length_opt,T)
        new_rho=rho*new_rho
        
        t2=time.clock()-t1-t0
        print "Step %d: M-Step I,  Time: %3.2f" % (steps,t2)
        
        # M-Step II a)
        T1=maximum_spanning_tree(links,number_of_nodes)
        
        # M-Step II b) 
        create_bifurcating_tree(1,T1,root)
        t3=time.clock()-t2-t1-t0
        print "Step %d: M-Step II, Time: %3.2f" % (steps,t3)
        
        #Likelihood 
        neighbors=neighbor_forks(root,T1)
        likelihood=likelihood_incomplete_data(neighbors[0],root,priors, D, T1)
        if(likelihood>likelihood_best):
            print "Likelihood(100 digits): %.100f" % likelihood
            best_tree=copy.deepcopy(T1)
            likelihood_best=likelihood
        #Update Tree
        T=copy.deepcopy(T1)
            
    if(best_tree!=None):
        return best_tree
    else:
        return T
  
'''
    Simulate multiple alignment
'''

def discrete_rv(probabilities):
    '''
    For a vector of probabilities it returns an index distributed by given probabilities 
    
    Output:
    integer out of [0,1,....,length(probabilities)-1]
    '''
    u=[probabilities[0]]
    for i in range(1,len(probabilities)):
        new_u=u[i-1]+probabilities[i]
        u.append(new_u)
    sample=np.random.random_sample()
    
    for l in range(len(probabilities)):
            if(sample<=u[l]):
                return l

def simulate_multiples_alignment(T,M,priors):
    '''
    Simulates a multiples alignment
    
    Input:
    T:                      bifurcating tree
    M:                      length of the alignment
    priors:                 prior probability for each nucleotide
    transition_matrix:      matrix with transition probabilties for the nucleotides
                
    Output:
    Dicitonary with a sequence of nuceleotides for each leaf of tree T  
    '''
    D={}
    N=len(T.nodes())
    nucleotides=list(priors)
    u=[priors[nucleotides[0]]]
    for l in range(1,len(nucleotides)):
        new_u=u[l-1]+priors[nucleotides[l]]
        u.append(new_u)
        
    # 1. Species 
    sequence=""
    for m in range(M):
        sample=np.random.random_sample()
        for l in range(len(nucleotides)):
            if(sample<u[l]):
                sequence=sequence+nucleotides[l]
                break
    D[str(N)]=sequence
    edges=T.dfs_edges(str(N))
    
    for edge in edges:
        if(edge[1] not in D):
            neighbor=""
            t=T.get_time(edge[0],edge[1])
            new_transition_matrix=transition_matrix**t
            for m in range(M):
                line=new_transition_matrix[states.get(D.get(str(edge[0]))[m]),0:]
                probabilities=line.tolist()[0]
                # adding random nucelotide to sequence 
                neighbor=neighbor+nucleotides[discrete_rv(probabilities)]
                D[str(edge[1])]=neighbor
    current_species=(N/2)+1
    D_current_day=create_subset(D,1,current_species,M)
    return D_current_day    
  

if __name__ == '__main__':
    example_graph=Graph()
 
