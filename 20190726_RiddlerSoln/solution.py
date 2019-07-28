import os, sys
import requests
import zipfile
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Agg')

def get_contiguous_states_list(data_path):
    df_cont_states = pd.read_csv('{0}/states.csv'.format(data_path))
    df_cont_states = df_cont_states[df_cont_states['CONTINENTAL'] == 'Y']    
    df_cont_states.drop('CONTINENTAL', axis=1, inplace=True)

    return df_cont_states

def get_contiguous_states(data_path):
    # Pull connectivity data
    connectivity_data_path = '{0}/contiguous-usa.dat'.format(data_path)
    if not os.path.exists(connectivity_data_path):
        url = 'https://www-cs-faculty.stanford.edu/~knuth/contiguous-usa.dat'
        print('Downloading boundary connections from {0}'.format(url))
        req = requests.get(url)
        contents = req.content
        with open(connectivity_data_path, 'wb') as f:
            f.write(contents)
    else:
        with open(connectivity_data_path, 'r') as f:
            contents = f.read()

    # Parse the contents to get 2 columns:
    #   data[0][:] = base state
    #   data[1][:] = border state
    data = [row.split(' ') for row in contents.split('\n')]
    data = [row for row in data if len(row) > 1]

    return data

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

    # create a new data frame with just the centroids
    df_cont_state_shps['centroid'] = df_cont_state_shps.centroid
    df_state_centroids = df_cont_state_shps.drop('geometry', axis=1)
    df_state_centroids = df_state_centroids.set_geometry('centroid')

    # Plot
    fig, ax = plt.subplots(figsize=(8,4))
    df_cont_state_shps.plot(ax=ax, color='white', edgecolor='black')
    df_state_centroids.plot(ax=ax, markersize=10, color='C1')
    plt.axis('off')
    plt.savefig('./test.png')
    
    
if __name__ == '__main__':
    
    data_path = './data'
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    # Get data
    get_contiguous_states_list(data_path)
    get_contiguous_states(data_path)
    get_contiguous_states_shapefile(data_path)
