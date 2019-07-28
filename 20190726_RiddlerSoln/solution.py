import os, sys
import requests
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import matplotlib
import matplotlib.pyplot as plt


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

    return df_cont_state_shps

def plot_connections(df_cont_state_shps, df_state_connxns, img_path=None):    
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
    fig, ax = plt.subplots()
    df_cont_state_shps.plot(ax=ax, color='white', edgecolor='black')
    df_plot_state_connxns.plot(ax=ax, zorder=1)
    df_state_centroids.plot(ax=ax, markersize=10, color='C1', zorder=2)    
    plt.axis('off')

    if img_path is not None:
        plt.savefig('{0}/state_border_connections.png'.format(img_path),
                    bbox_inches='tight', dpi=400)
        

    return fig, ax, df_plot_state_connxns
    
    
if __name__ == '__main__':
    
    data_path = './data'
    img_path  = './img'
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    if not os.path.exists(img_path):
        os.mkdir(img_path)

    # Get data
    df_cont_states     = get_contiguous_states_list(data_path)
    df_state_connxns   = get_contiguous_state_connections(data_path)
    df_cont_state_shps = get_contiguous_states_shapefile(data_path)

    # plot
    if not os.path.exists('{0}/state_border_connections.png'.format(img_path)):
        fig, ax, df_plot_state_connxns = plot_connections(df_cont_state_shps, df_state_connxns, img_path)
