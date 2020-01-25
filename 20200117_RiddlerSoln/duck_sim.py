import numpy as np

dir_dict = {'N' : np.array([ 0,  1],  dtype='int'),
            'E' : np.array([ 1,  0],  dtype='int'),
            'S' : np.array([ 0, -1], dtype='int'),
            'W' : np.array([-1,  0], dtype='int')}

def duck_update(p, grid_size = 3):
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

    jump_dir = np.random.choice(cand_dirs)
    v = dir_dict[jump_dir]
    
    return p + v

def update_all_ducks(duck_positions, grid_size):
    for i in range(duck_positions.shape[0]):
        duck_positions[i,:] = duck_update(duck_positions[i,:], grid_size)

    return duck_positions

def ducks_at_middle(duck_positions, middle):
    return np.all(duck_positions == middle)

def ducks_together(duck_positions):
    return np.equal.reduce(np.all(duck_positions == duck_positions[0, :]), axis=0)

def duck_simulator(n_ducks = 2, grid_size = 3):
    middle = (grid_size - 1) // 2

    duck_positions = np.repeat(
        np.array([middle, middle], dtype='int').reshape(1,2),
        repeats = n_ducks,
        axis = 0            
    )
    
    duck_positions = update_all_ducks(duck_positions, grid_size)
    n_moves = 1 
    
    while not(ducks_together(duck_positions)):
        duck_positions = update_all_ducks(duck_positions, grid_size)
        n_moves += 1

    return n_moves


def duck_simulator_repeated(n_ducks = 2, grid_size = 3, n_sim = 1):
    if grid_size % 2 != 1:
        return -1

    n_moves_sim = np.zeros(n_sim)
    
    for k in range(n_sim):
        n_moves_sim[k] = duck_simulator(n_ducks = n_ducks, grid_size = grid_size)

    avg_n_moves = np.mean(n_moves_sim)
    
    return avg_n_moves, n_moves_sim
    
if __name__ == '__main__':
    grid_size = 5
    n_ducks = 3
    n_sim = 10000

    mean, simulated = duck_simulator_repeated(n_ducks, grid_size, n_sim)

    print(mean)
