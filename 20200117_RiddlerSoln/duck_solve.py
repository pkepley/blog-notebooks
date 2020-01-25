import numpy as np
from scipy.sparse import dok_matrix, issparse
from scipy.sparse.linalg import lsqr, svds, spsolve
from scipy.linalg import null_space, lstsq, solve
from scipy.optimize import lsq_linear, minimize, nnls, linprog
import matplotlib.pyplot as plt
import networkx as nx

dir_dict = {'N' : np.array([ 0,  1],  dtype='int'),
            'E' : np.array([ 1,  0],  dtype='int'),
            'S' : np.array([ 0, -1], dtype='int'),
            'W' : np.array([-1,  0], dtype='int')}

def cell_neighbors(p, grid_size):
    e2 = grid_size - 1    
    cand_dirs = ['N', 'E', 'S', 'W']

    # Top or bottom
    if p[1] == e2:
        cand_dirs.remove('N')
        
    elif p[1] == 0:
        cand_dirs.remove('S')

    # Left or right
    if p[0] == 0:
        cand_dirs.remove('W')
        
    elif p[0] == e2:
        cand_dirs.remove('E')
    
    return [p + dir_dict[cd] for cd in cand_dirs]

def k_to_pos(k, grid_size):
    x = k % grid_size
    y = (k - x) // grid_size
    pos = np.array([x,y], dtype='int')
    
    return pos

def pos_to_k(pos, grid_size):
    return grid_size * pos[1] + pos[0]

def single_duck_transition_matrix(grid_size = 3):
    # e2 is the size of the rightmost edge / topmost edge: 1 less than grid_size
    # since leftmost / bottom-most are at 0
    e2 = grid_size - 1

    # Transition matrix
    P = dok_matrix((grid_size**2, grid_size**2), dtype='float')

    for y in range(grid_size):
        for x in range(grid_size):
            # Determine index of pos = (x,y)
            pos = np.array([x,y], dtype = 'int')            
            k = pos_to_k(pos, grid_size)

            # Compute neighbors of pos and uniform transition prob to neighbors
            pos_neighbor_pos = cell_neighbors(pos, grid_size)
            pos_neighbor_ks  = [pos_to_k(n, grid_size) for n in pos_neighbor_pos]
            transition_prob = 1.0 / len(pos_neighbor_ks)

            # Fill the transition matrix
            for k_neigh in pos_neighbor_ks:
                P[k, k_neigh] = transition_prob

    return P

def idx_to_k_tuple(idx, grid_size, n_ducks):
    k_tuple = np.zeros(n_ducks, dtype = 'int')

    for i in range(n_ducks-1, -1, -1):
        k_tuple[i] = idx % grid_size**2
        idx //= grid_size**2

    return k_tuple


def k_tuple_to_idx(k_tuple, grid_size, n_ducks):
    idx = 0

    for i in range(0, n_ducks):
        idx *= (grid_size**2)
        idx += k_tuple[i]

    return idx

                        
def multi_duck_transition_matrix(grid_size = 3, n_ducks = 2):
    if n_ducks == 1:
        P_n = single_duck_transition_matrix(grid_size)

    else:
        # N-1 duck matrix
        P_n_1 = multi_duck_transition_matrix(grid_size, n_ducks - 1)
        items_n_1 = list(P_n_1.items())

        # 1 duck matrix
        P_1   = single_duck_transition_matrix(grid_size)
        items_1 = list(P_1.items())

        # N duck matrix
        n_k   = grid_size**2
        P_n   = dok_matrix((n_k**n_ducks, n_k**n_ducks), dtype='float')

        for elt_n_1 in items_n_1:
            # See assertions (*) for meaning of r and q
            r,q  = elt_n_1[0]
            a    = elt_n_1[1]
            
            for elt_1 in items_1:
                # See assertions (*) for meaning of t and u
                t,u = elt_1[0]
                b   = elt_1[1]

                k1 = (n_k)**(n_ducks - 1) * t + r
                k2 = (n_k)**(n_ducks - 1) * u + q                

                # (*) Assertions for r,q,t,u
                assert(k1  % n_k**(n_ducks-1) == r)
                assert(k2  % n_k**(n_ducks-1) == q)
                assert(k1 // n_k**(n_ducks-1) == t)
                assert(k2 // n_k**(n_ducks-1) == u)
                
                P_n[k1, k2] = a * b
                    
    return P_n

def modified_multi_duck_transition_matrix(grid_size = 3, n_ducks = 2, P_n = None):
    if P_n == None:
        P_n = multi_duck_transition_matrix(grid_size, n_ducks)
        
    Q_n = P_n[:,:]
        
    nz_entries = list(P_n.keys())
        
    for k in range(grid_size**2):
        k_tuple = np.repeat(k, n_ducks)
        idx = k_tuple_to_idx(k_tuple, grid_size, n_ducks)
        idx_row = [ij for ij in nz_entries if ij[0] == idx]

        for ij in idx_row:
            Q_n.pop(ij)

        Q_n[(idx, idx)] = 1
            
    return Q_n


def hitting_time_eqn_matrix(grid_size = 3, n_ducks = 2, Q_n = None):
    if Q_n == None:    
        Q_n = modified_multi_duck_transition_matrix(grid_size, n_ducks)

    # Define idx_middle, the index where all ducks are together
    # in the middle cell    
    middle = grid_size // 2
    middle_cell = pos_to_k(np.array([middle, middle], dtype='int'), grid_size)    
    k_middle = np.repeat(middle_cell, n_ducks).astype('int')
    idx_middle = k_tuple_to_idx(k_middle, grid_size, n_ducks) 

    # Some states will be unreachable if the ducks are all together in the
    # beginning. So get networkx graph of Q_n to find all states which are
    # reachable from idx_middle
    if issparse(Q_n):    
        g = nx.convert_matrix.from_scipy_sparse_matrix(Q_n)
    else:
        g = nx.convert_matrix.from_numpy_matrix(Q_n)        

    # idxs connected to middle cell
    g = g.to_undirected()
    nodes = nx.shortest_path(g, idx_middle).keys()

    # Flip on all idxs for cells in middle cell connected component
    # (we want to keep these)
    n_k  = grid_size**2
    mask = np.zeros(n_k**n_ducks, dtype='bool')
    for i in list(nodes):
         mask[i] = True

    # Construct a mask to drop absorbing states:
    # Indices of the absorbing states:
    k_tuples_A = [np.repeat(k, n_ducks) for k in range(grid_size**2)]
    idxs_A     = [k_tuple_to_idx(k_tuple, grid_size, n_ducks) for
                  k_tuple in k_tuples_A]

    # Add absorbing states to mask (we don't want to keep these)
    for i in idxs_A:
         mask[i] = False

    # Idx inverse
    idxH_to_Q = np.arange(len(mask), dtype='int')[mask]
         
    # Apply mask to Q_n. dok seems finnicky, so do it in 2 steps:
    Q_red = Q_n[:,:]
    Q_red = Q_red[mask, :]
    Q_red = Q_red[:, mask]
    
    # Matrix will be H = I - P_red
    I = dok_matrix(Q_red.shape, dtype='float')    
    I.setdiag(1.0)

    # Set H
    H = I - Q_red
    
    return H, idxH_to_Q


def solve_hitting_time_eqn(grid_size = 3, n_ducks = 2, H = None):
    if H is None:
        H, idxH_to_Q = hitting_time_eqn_matrix(grid_size, n_ducks)

    # Min-norm solution to H k = 1
    rhs = np.ones((H.shape[0], 1))

    if issparse(H):
        #rslt = lsqr (H, rhs)
        kA = spsolve(H, rhs)        
    else:
        #rslt = lstsq(H, rhs)
        kA = solve(H, rhs)        

    return kA.reshape((-1,1))

def solve_riddler(grid_size = 3, n_ducks = 2, H = None, idxH_to_Q = 
                  None, P_n = None):
    
    if H is None or idxH_to_Q is None:
        H, idxH_to_Q = hitting_time_eqn_matrix(grid_size, n_ducks)

    if P_n == None:
        P_n = multi_duck_transition_matrix(grid_size, n_ducks)

    # Define idx_middle, the index where all ducks are together
    # in the middle cell
    middle = grid_size // 2    
    middle_cell = pos_to_k(np.array([middle, middle], dtype='int'), grid_size)    
    k_middle = np.repeat(middle_cell, n_ducks).astype('int')
    idx_middle = k_tuple_to_idx(k_middle, grid_size, n_ducks) 
        
    # Compute k_red, the mean hitting time from each index
    mean_hitting_time_red = solve_hitting_time_eqn(grid_size, n_ducks, H)
    mean_hitting_time  = np.zeros(P_n.shape[0], dtype='float')

    for i in range(len(idxH_to_Q)):
        mean_hitting_time[idxH_to_Q[i]] = mean_hitting_time_red[i]

    # Distro at first step after all ducks leave middle cell
    initial_distro = P_n[idx_middle, :].toarray()

    # The mean time for the to meet again, given that they all started
    # at the middle cell is 1 more than the mean time averaged over the
    # possible starting positions at first step
    mean_time_to_absorption = 1 + np.dot(mean_hitting_time.flatten(),
                                         initial_distro.flatten())
        
    return mean_time_to_absorption
    

if __name__ == '__main__':    
    grid_size = 5
    n_ducks = 3
    
    P = multi_duck_transition_matrix(grid_size, n_ducks)
    Q = modified_multi_duck_transition_matrix(grid_size, n_ducks, P_n = P)
    H, idxH_to_Q = hitting_time_eqn_matrix(grid_size, n_ducks, Q_n =Q)
    kA = solve_hitting_time_eqn(grid_size, n_ducks, H)

    mean_time = solve_riddler(grid_size, n_ducks, H, idxH_to_Q, P)
    print(mean_time)
