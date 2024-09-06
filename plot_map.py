import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import geopandas as gpd
'''
https://www.naturalearthdata.com/downloads/110m-cultural-vectors/

https://jan-46106.medium.com/plotting-maps-with-european-data-in-python-part-i-decd83837de4
'''


# load the low resolution world map
def plot_map_world_lowres(extra_data: pd.DataFrame, area: str = None):
    '''
    You can set area to be
    1. No specify: no filtering
    2. 'in data': will use the column 'NAME' in the extra_data to filter the countries
    3. name of continent, eg. "Europe": will filter to include countries in that continent
    '''
    cols_to_keep = [
        'CONTINENT',
        # 'REGION_UN',
        # 'SUBREGION',
        # 'REGION_WB'
        'NAME',
        'ISO_A3',
        'ISO_N3',
        # 'LABEL_X', # lon
        # 'LABEL_Y', # lat
        'geometry'
    ]

    countries = gpd.read_file('./data/ne_110m_admin_0_countries/')
    if not area:
        cond = countries['NAME'] == countries['NAME']
    elif area == 'in data':
        cond = countries['NAME'].isin(extra_data['NAME'])
    else:
        cond = countries['CONTINENT'] == area

    countries = countries.loc[cond, cols_to_keep]

    print(sorted(countries['NAME']))

    # print(set(extra_data['NAME']) - set(countries['NAME']))
    # print(set(countries['NAME']) - set(extra_data['NAME']))

def plot_map_europe(extra_data: pd.DataFrame, area: str = None):
    '''
    You can set area to be
    1. No specify: no filtering
    2. 'in data': will use the column 'NAME' in the extra_data to filter the countries
    3. name of continent, eg. "Europe": will filter to include countries in that continent
    '''
    cols_to_keep = [
        'CONTINENT',
        # 'REGION_UN',
        # 'SUBREGION',
        # 'REGION_WB'
        'NAME',
        'ISO_A3',
        'ISO_N3',
        # 'LABEL_X', # lon
        # 'LABEL_Y', # lat
        'geometry'
    ]

    countries = gpd.read_file('./data/ne_110m_admin_0_countries/')
    if not area:
        cond = countries['NAME'] == countries['NAME']
    elif area == 'in data':
        cond = countries['NAME'].isin(extra_data['NAME'])
    else:
        cond = countries['CONTINENT'] == area

    countries = countries.loc[cond, cols_to_keep]

    print(sorted(countries['NAME']))

def generate_color_col(
        srs: pd.Series,
        levels: int, 
        colors: list = ['blue', 'white', 'red']
    ):
    """
    Generates a custom blue-to-red colormap with the specified number of levels.

    Parameters:
        levels (int): Number of discrete levels in the colormap.

    Returns:
        LinearSegmentedColormap: The generated colormap.
    """
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_color", colors, N=levels)
    
    norm = mcolors.Normalize(vmin=0, vmax=100)
    return [mcolors.to_hex(cmap(norm(p))) for p in srs]


def task_europe_friendliness():
    '''
    ['country', 'year', 'unfriendly_cnt', 'friendly_cnt',
    'friendly_index_net']
    '''
    df = pd.read_stata("./data/friendly_index_net.dta")
    cols_to_keep = ['country', 'year', 'friendly_index_net']
    df = df[cols_to_keep]
    df['year'] = df['year'].astype('int32')

    name_map = pd.read_csv("./data/eu_country_name_mapping.csv")

    df = pd.merge(df, name_map, on='country', how='left')

    df['color'] = generate_color_col(
        srs=df['friendly_index_net'].rank(pct=True) * 100,
        levels=10
    )

    cur_year = 2017

    df = df[df['year'] == cur_year]
    print(f"Countries in {cur_year}: \n{df['NAME']}")
    plot_map_europe(df, area='in data')
    # 全部加總的顏色要另外等加總完之後再做
    
    return 

def main():
    task_europe_friendliness()


if __name__ == "__main__":
    main()