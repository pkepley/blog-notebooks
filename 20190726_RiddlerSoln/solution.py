import os, sys
import requests
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from simulated_annealing import simulated_annealing

def get_contiguous_states_list(data_path):
    df_cont_states = pd.read_csv('{0}/states.csv'.format(data_path))
    df_cont_states = df_cont_states[df_cont_states['CONTINENTAL'] == 'Y']    
    df_cont_states.drop('CONTINENTAL', axis=1, inplace=True)

    return df_cont_states

def get_contiguous_state_connections(data_path):
    # Pull connectivity data
    connectivity_data_path = '{0}/contiguous-usa.dat'.format(data_path)
    if not os.path.exists(connectivity_data_path):
        url = 'https://www-cs-faculty.stanford.edu/~knuth/contiguous-usa.dat'
        print('Downloading boundary connections from {0}'.format(url))
        req = requests.get(url)
        contents = req.content
        with open(connectivity_data_path, 'wb') as f:
            f.write(contents)
        contents = contents.decode('utf-8')
    else:
        with open(connectivity_data_path, 'r') as f:
            contents = f.read()

    # Parse the contents to get 2 columns:
    state_connxns = [row.split(' ') for row in contents.split('\n')]
    state_connxns = np.array([row for row in state_connxns if len(row) > 1])
    df_state_connxns = pd.DataFrame(state_connxns, columns=['BASE_STATE', 'BORDER_STATE'])

    df_cont_states = get_contiguous_states_list(data_path)
    df_state_connxns = pd.merge(df_state_connxns, df_cont_states, how='inner',
                                left_on = 'BASE_STATE', right_on = 'STATE')
    df_state_connxns.drop(columns='STATE', inplace=True)
    df_state_connxns = pd.merge(df_state_connxns, df_cont_states, how='inner',
                                left_on = 'BORDER_STATE', right_on = 'STATE')
    df_state_connxns.drop(columns='STATE', inplace=True)

    # Ensure it's sorted
    df_state_connxns.sort_values(by = ['BASE_STATE', 'BORDER_STATE'], axis=0, ascending=True,
                                 inplace=True)

    return df_state_connxns

def get_contiguous_states_shapefile(data_path):
    # Pull contiguous states shapefile from NWS
    cont_states_shpfl_zip_path = '{0}/s_11au16.zip'.format(data_path)
    cont_states_shpfl_path = '{0}/s_11au16.shp'.format(data_path)
    
    if not os.path.exists(cont_states_shpfl_zip_path):
        url = 'https://www.weather.gov/source/gis/Shapefiles/County/s_11au16.zip'
        req = requests.get(url, stream=True)
        print('Downloading shapefile from {0}'.format(url))
        with open(cont_states_shpfl_zip_path, 'wb') as f:
            for chunk in req.iter_content(chunk_size=512):
                if chunk:
                    f.write(chunk)
                    
    if not os.path.exists(cont_states_shpfl_path):
        print('Extracting contents from {0} to {1}'.format(cont_states_shpfl_zip_path,
                                                           data_path))        
        with zipfile.ZipFile(cont_states_shpfl_zip_path) as zip_ref:
            zip_ref.extractall(data_path)


    # get the list of continental states
    df_cont_states = get_contiguous_states_list(data_path)
            
    # read the shapefile of all states etc
    df_cont_state_shps = gpd.read_file(cont_states_shpfl_path)
    df_cont_state_shps = df_cont_state_shps[abs(df_cont_state_shps['LAT']) > 1] # I was getting two Marylands, this hack avoids it

    # inner join on the state/name to get only the continental states
    df_cont_state_shps = pd.merge(df_cont_state_shps, df_cont_states,
                                  on = ['STATE', 'NAME'], how = 'inner')

    # sort the data
    df_cont_state_shps.sort_values(by = ['STATE'], axis=0, ascending=True, inplace=True)

    return df_cont_state_shps

def plot_connections(df_cont_state_shps, df_state_connxns, ax=None, img_path=None):    
    # create a new data frame with just the centroids
    df_cont_state_shps['centroid'] = df_cont_state_shps.centroid
    df_state_centroids = df_cont_state_shps.drop('geometry', axis=1)
    df_state_centroids = df_state_centroids.set_geometry('centroid')

    # plot the connections
    df_plot_state_connxns = pd.merge(df_state_connxns,
                                     df_state_centroids,
                                     how = 'left',
                                     left_on = ['BASE_STATE'],
                                     right_on = ['STATE'],
                                     suffixes=('','_base_state'))
    df_plot_state_connxns = pd.merge(df_plot_state_connxns,
                                     df_state_centroids,
                                     how = 'left',
                                     left_on = ['BORDER_STATE'],
                                     right_on = ['STATE'],
                                     suffixes=('','_border_state'))
    df_plot_state_connxns.drop(['STATE', 'FIPS', 'LON', 'LAT', 'FIPS_border_state', 'LON_border_state', 'LAT_border_state', 'STATE_border_state'],
                               axis=1, inplace=True)
    df_plot_state_connxns = df_plot_state_connxns.rename(columns = {'centroid' : 'centroid_base_state', 'NAME' : 'NAME_border_state'})
    df_plot_state_connxns['connecting_line'] = df_plot_state_connxns.apply(lambda x: LineString([x['centroid_base_state'], x['centroid_border_state']]), axis=1)
    df_plot_state_connxns = gpd.GeoDataFrame(df_plot_state_connxns)
    df_plot_state_connxns['geometry'] = df_plot_state_connxns['connecting_line']
    df_plot_state_connxns.set_geometry('geometry')
    
    # Plot
    if ax is None:
        fig, ax = plt.subplots()
        
    df_cont_state_shps.plot(ax=ax, color='white', edgecolor='black')
    df_plot_state_connxns.plot(ax=ax, zorder=1)
    df_state_centroids.plot(ax=ax, markersize=10, color='C1', zorder=2)    
    plt.axis('off')

    if img_path is not None:
        plt.savefig('{0}/state_border_connections.png'.format(img_path),
                    bbox_inches='tight', dpi=400)
        

    return ax, df_plot_state_connxns


def generate_state_graph(df_state_connxns):
    # Create a graph and add the states as nodes
    state_graph = nx.Graph()
    
    # Add the edges
    for row in df_state_connxns.iterrows():
        edge = (row[1]['BASE_STATE'], row[1]['BORDER_STATE'])
        state_graph.add_edge(row[1]['BASE_STATE'], row[1]['BORDER_STATE'])

    return state_graph

        
def generate_graph_distances(state_graph, n_states):
    # Matrix to hold distances
    distances = np.zeros((n_states, n_states))

    # Compute the distances
    for i in range(n_states):
        for j in range(i, n_states):
            distances[i,j] = nx.shortest_path_length(state_graph, states[i], states[j])
            distances[j,i] = distances[i, j]

    return distances

def generate_graph_pos(state_graph):
    # Assign positions to the graph for displaying network without actual physical
    # locations
    pos = nx.spring_layout(state_graph, seed=521235)
    
    return pos

def plot_distances(distances, ax=None, img_path=None):
    if ax is None:
        fig, ax = plt.subplots()

    # plot the distance matrix and add a colorbar
    im = ax.imshow(distances)
    plt.colorbar(im)

    # set up the axis labels
    ax.set_xticks(np.arange(n_states))
    ax.set_xticklabels(states, fontsize=6, rotation=90)
    ax.set_yticks(np.arange(n_states))
    ax.set_yticklabels(states, fontsize=6)

    # show the image
    plt.show(block=False)

    # save the image if we have an img_path to save at
    if img_path is not None:
        plt.savefig('{0}/distances.png'.format(img_path), bbox_inches='tight', dpi=400)

    return ax

    
def step_plotter_graph(city_order, dist, step_t, states, pos, state_graph,
                       ax=None, img_path=None):
    n_cities = len(city_order)
    points = [pos[states[i]] for i in city_order]
    points = np.array(points)
    ax.clear()
    nx.draw(state_graph, pos = pos, with_labels=True, ax=ax)
    ax.plot(points[:, 0], points[:, 1], c = 'C1', linewidth=3)
    plt.draw()
    plt.pause(0.25)

    # Output image to file
    if img_path is not None:
        plt.savefig('{0}/step_graphviz_{1:04d}.png'.format(img_path, step_t),
                    bbox_inches='tight', dpi=200)
    return ax

    
def step_plotter_states(city_order, dist, step_t, states, centroids, df_cont_state_shps, df_state_connxns,
                        ax=None, img_path=None):

    if ax is None:
        fig, ax = plt.subplots()
    
    n_cities = len(city_order)
    points = [centroids.loc[states[i]].values for i in city_order]
    points = np.array(points)

    for c in ax.lines:
        if c.get_gid() in ['path', 'points']:
            c.remove()

    # Plot the path and highlight the centroids
    ax.plot(points[:, 0], points[:, 1],    c = 'C3', gid = 'path')
    ax.scatter(points[:, 0], points[:, 1], c = 'k',  s=16, zorder=100, gid = 'points')

    # Set the title
    ax.set_title('Step {0}: Best path has length {1}'.format(step_t, int(dist)))    
    plt.draw()
    plt.pause(.25)

    # Output image to file
    if img_path is not None:
        plt.savefig('{0}/step_stateviz_{1:04d}.png'.format(img_path, step_t),
                    bbox_inches='tight', dpi=200)

    return ax
    
if __name__ == '__main__':
    # Where we will save data
    data_path = './data'
    img_path  = './img'

    # What type of graphics we want
    graphics_type = 'state_plot'
    
    # Create loctions to save data if they don't exist
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    
    # Get data
    df_cont_states     = get_contiguous_states_list(data_path)
    df_state_connxns   = get_contiguous_state_connections(data_path)
    df_cont_state_shps = get_contiguous_states_shapefile(data_path)
    states = sorted(list(df_cont_states['STATE'].values))
    n_states = len(states)
    
    # Generate state graph and distances
    state_graph = generate_state_graph(df_state_connxns)
    distances   = generate_graph_distances(state_graph, n_states)
        
    # Make plots interactive for now
    plt.ion()

    # Plot border connections. Only save if we haven't created it yet.
    fig1, ax1 = plt.subplots()
    ax1, df_plot_state_connxns = plot_connections(df_cont_state_shps, df_state_connxns, ax = ax1, img_path = img_path)
    plt.draw()
    plt.pause(.25)
    
    # Plot the distance matrix. Only save if we haven't created it yet.
    fig2,  ax2 = plt.subplots()            
    plot_distances(distances, ax=ax2, img_path=img_path)
    plt.draw()
    plt.pause(.25)    

    # Set up some graphical output if a plot of state locations is desired
    if graphics_type == 'state_plot':
        # State centroids for plotting
        centroids = df_cont_state_shps[['STATE', 'LON', 'LAT']].set_index('STATE')
        
        # Plot the state graph for the background of the path plots
        fig3, ax3 = plt.subplots()
        plot_connections(df_cont_state_shps, df_state_connxns, ax=ax3)
        plt.show(block=False)   
        step_logger = lambda city_order, dist, step_t : step_plotter_states(city_order,
                                                                 dist,
                                                                 step_t,
                                                                 states=states,
                                                                 centroids = centroids,
                                                                 df_cont_state_shps = df_cont_state_shps,
                                                                 df_state_connxns = df_state_connxns,
                                                                 ax = ax3,
                                                                 img_path = img_path)
        
    # Set up some graphical output if a plot of the network topology
    # with randomly seeded positions will suffice
    else:
        pos = generate_graph_pos(state_graph)
        # Plot the state graph for the background of the path plots
        fig3, ax3 = plt.subplots()
        nx.draw(state_graph, pos = pos, with_labels=True, ax=ax3)        
        plt.show(block=False)
        step_logger = lambda city_order, dist, step_t : step_plotter_graph(city_order,
                                                                           dist,
                                                                           step_t,
                                                                           states=states,
                                                                           pos=pos,
                                                                           state_graph = state_graph,
                                                                           ax = ax3,
                                                                           img_path = img_path)
        
    # Set a seed for reproducibility
    np.random.seed(12345)

    # Run the simulated annealing algorithm
    best_order, best_dist = simulated_annealing(
        cities = np.arange(n_states),
        distances = distances,
        p_reverse = 0.5,
        max_T_steps = 100,
        T_start = 0.75,
        t_reduce_factor = 0.9,
        max_k_steps = 100 * n_states,
        max_accepted = 10 * n_states,
        stopping_dist = 47,
        step_logger = step_logger
    )

    # Print out the best path found
    print('\nBest Path Found:')
    print([states[i] for i in best_order])

    # write best path found to file
    with open('{0}/best_path_found.txt'.format(data_path), 'w') as file:
        for i in best_order:
            file.write("{0}\n".format(states[i]))

    #
    try:
        if graphics_type == 'state_plot':
            os.system('convert -delay 50 {0}/step_stateviz_*.png -delay 100 -loop 0 {0}/sa_stateviz_solution.gif'.format(img_path))
        else:
            os.system('convert -delay 50 {0}/step_graphviz_*.png -delay 100 -loop 0 {0}/sa_graphviz_solution.gif'.format(img_path))            
    except:
        print('Something went wrong making the gif. This step requires Imagemagick to be installed and on the path.')
            
    # Clean up matplotlib
    plt.ioff()
    plt.show()
        
