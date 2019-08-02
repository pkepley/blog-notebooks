import numpy as np
import matplotlib.pyplot as plt

def path_length(city_order, distances):
    n_cities = len(city_order)
    path_length = 0
    
    for i in range(n_cities - 1):
        path_length += distances[city_order[i], city_order[i+1]]
        
    return path_length

def swap(idx_1, idx_2, city_order):
    assert idx_1 < idx_2
    
    city_1 = city_order[idx_1]
    city_2 = city_order[idx_2]
    
    new_city_order = city_order[:]
    new_city_order[idx_2] = city_1
    new_city_order[idx_1] = city_2

    return new_city_order

def reverse_path(idx_1, idx_2, city_order):
    assert idx_1 < idx_2

    new_city_order = city_order[:]
    new_city_order[idx_1:(idx_2+1)] = reversed(new_city_order[idx_1:(idx_2+1)])

    return new_city_order
    
def simulated_annealing(cities, distances, max_T_steps, T_start, t_reduce_factor, p_reverse,
                        max_k_steps, max_accepted, stopping_dist = 0, step_logger=None):
    # Initialize problem
    T = T_start
    n_cities = len(cities)
    city_order = list(np.random.permutation(np.arange(n_cities)))
    stopping_dist = stopping_dist

    # Initialize objective and optimal candidates
    dist = path_length(city_order, distances)
    best_dist_found  = dist
    best_order_found = city_order[:]

    # Define a throw-away logger if none provided
    if step_logger is None:
        step_logger = (lambda *args: None)

    for step_t in range(max_T_steps):
        n_accepted = 0
        
        for k in range(max_k_steps):
            # generate indicies to swap or reverse through
            idx_1, idx_2 = np.random.choice(np.arange(n_cities), 2, replace=False)            
            if idx_1 > idx_2:
                idx_1, idx_2 = idx_2, idx_1

            # try a reversal
            if p_reverse > np.random.rand():
                tmp_dist = path_length(reverse_path(idx_1, idx_2, city_order[:]), distances)                
                de = tmp_dist - dist

                if de < 0 or np.exp(-de/T) > np.random.rand():
                    n_accepted += 1
                    dist = tmp_dist
                    city_order = swap(idx_1, idx_2, city_order)
                    

            # try a transposition
            else:
                tmp_dist = path_length(swap(idx_1, idx_2, city_order[:]), distances)
                de = tmp_dist - dist
                
                if de < 0 or np.exp(-de/T) > np.random.rand():
                    n_accepted += 1
                    dist = tmp_dist
                    city_order = swap(idx_1, idx_2, city_order)

            if dist < best_dist_found:
                best_dist_found  = dist
                best_order_found = city_order[:]                                                        

            if n_accepted > max_accepted:                
                break
            
        print("T=%10.5f , distance= %10.5f , accepted steps= %d" % (T, dist, n_accepted))
        T *= t_reduce_factor

        step_logger(city_order, best_dist_found, step_t)

        if dist <= stopping_dist:
            break

    step_logger(best_order_found, best_dist_found, step_t)        

    return best_order_found, best_dist_found
