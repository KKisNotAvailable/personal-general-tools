import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import geopandas as gpd
from PIL import Image
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")
'''
https://www.naturalearthdata.com/downloads/110m-cultural-vectors/

https://jan-46106.medium.com/plotting-maps-with-european-data-in-python-part-i-decd83837de4
'''

OUTPUT_PATH = './output/'


# load the low resolution world map
def plot_map_world(extra_data: pd.DataFrame = None, area: str = None):
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

    # countries = gpd.read_file('./data/ne_110m_admin_0_countries/') # low res
    # using map_units and select Europe would get the map we want!
    # (guess it just seperates France oversea territories from mainland)
    countries = gpd.read_file('./data/ne_10m_admin_0_map_units/')  # high res

    if not area:
        cond = countries['NAME'] == countries['NAME']
    else:
        cond = countries['CONTINENT'] == area

    # Cyprus need to be included (for current need)
    # and exclude Russia (cuz its too large)
    if area == "Europe":
        cond = (countries['NAME'] != "Russia") & cond |\
            (countries['NAME'] == "Cyprus")

    countries = countries.loc[cond, cols_to_keep]
    # countries.to_csv("./data/test.csv")

    countries[['NAME', 'ISO_A3', 'ISO_N3']].to_csv("./data/test.csv")

    # sizex, sizey = 12, 9
    # ax = countries.plot(figsize=(sizex, sizey), color="gray")
    # # gdfs['LB'].plot(figsize=(sizex, sizey), ax=ax, color="blue")

    # plt.axis("off")
    # plt.show()


def plot_map_europe(
        extra_data: pd.DataFrame = None, 
        cmap=None, 
        bound=None,
        title: str = None
    ):
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
        # 'ISO_A3',
        # 'ISO_N3',
        # 'LABEL_X', # lon
        # 'LABEL_Y', # lat
        'geometry'
    ]

    # 2. get data of other countries
    # Include Cyprus (for current need) and exclude Russia (too large)
    countries = gpd.read_file('./data/ne_10m_admin_0_countries/')  # high res

    cond = (countries['NAME'] != "Russia") &\
        (countries['CONTINENT'] == "Europe") |\
        (countries['NAME'] == "Cyprus")

    countries = countries.loc[cond, cols_to_keep].reset_index(drop=True)
    countries = pd.merge(countries, extra_data, on='NAME', how='left')

    # countries['color'].fillna('#808080', inplace=True) # will not work in future
    countries.fillna({'color': '#808080'}, inplace=True)

    sizex, sizey = 12, 9
    ax = countries.plot(
        figsize=(sizex, sizey), 
        color=countries.color,
        edgecolor='black',        # Color of the edges
        linewidth=0.5             # Width of the edges
    )

    # actually, setting the display limit solves the territory problem
    ax.set_xlim([-25, 45])  # Longitude
    ax.set_ylim([30, 72])   # Latitude

    ax.set_aspect(1.5)  # adjust the display ratio

    plt.gcf().set_facecolor('dodgerblue') # set background color

    # ======================
    # add the country labels
    # ======================
    # extra_data = extra_data.sort_values(
    #     by='friendly_index_net', ascending=False, inplace=False
    # )
    # lengend_txt = []
    # for i, c in enumerate(extra_data['NAME']):
    #     cur_row = countries[countries['NAME'] == c]
    #     sn = i + 1
    #     lengend_txt.append(f"{sn}: {c}")
    #     plt.text(
    #         cur_row['geometry'].centroid.x,
    #         cur_row['geometry'].centroid.y,
    #         str(sn), ha='center', color='yellowgreen',
    #         fontsize=8, weight='bold'
    #     )
    # lengend_txt = '\n'.join(lengend_txt)
    # plt.text(
    #     1.1, 0.5, lengend_txt, transform=ax.transAxes, 
    #     fontsize=10, verticalalignment='center'
    # )
    # # Adjust layout to make room for the legend
    # plt.subplots_adjust(right=0.8)

    # =============
    # add color bar
    # =============
    sm = plt.cm.ScalarMappable(cmap=cmap)
    cbar = plt.colorbar(
        sm, ax=ax, fraction=0.035, pad=0.04,
        orientation='vertical'
    )
    np_bound = np.array(bound)
    normalized = (np_bound - min(np_bound)) / (max(np_bound) - min(np_bound))
    cbar.set_ticks(normalized)
    cbar.set_ticklabels(
        np.round(bound, 0).astype(int), 
        ha='right', # horizontal alignment
        x=2.0 # the starting point of label, =0 will be overlapping with the colorbar
    )

    # ticks_lbs = [
    #     'Very Low', 'Low', 'Neutral Low',
    #     'Neutral High', 'High', 'Very High'
    # ]
    # cbar.set_ticklabels(ticks_lbs)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.set_position([0.25, 0.2, 0.03, 0.38])  # (left, bottom, width, height)

    cbar.set_label('Friendly Index')
    cbar.ax.yaxis.set_label_position('left')

    if title:
        plt.title(title, fontsize=16, weight='bold')

    plt.axis("off")
    # plt.savefig(OUTPUT_PATH + f'{title}.svg')
    plt.show()


def generate_color_col(
    srs: pd.Series,
    levels: int = None,
    step: int = 1,
    colors: list = ['red', 'white', 'blue']
):
    """
    Generates a custom blue-to-red colormap with the specified number of levels.

    Parameters:
        levels (int): Number of discrete levels in the colormap.

    Returns:
        LinearSegmentedColormap: The generated colormap.
    """
    mv = max(abs(srs))
    if not levels and step:
        t = 2 if srs.lt(0).any() else 1
        levels = np.ceil(mv / step) * t

    levels = int(levels)
    bound = np.linspace(-mv, mv, levels+1)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_color", colors, N=levels
    )
    norm = mcolors.Normalize(vmin=-mv, vmax=mv)
    return [mcolors.to_hex(cmap(norm(p))) for p in srs], cmap, bound


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

    df['color'], cmap, bound = generate_color_col(
        srs=df['friendly_index_net'],
        step=3
    )

    y_list = sorted(set(df['year']))
    # y_list = [2017]

    for cur_year in y_list:
        # not all countries are included in each yaer
        cols = ['NAME', 'friendly_index_net', 'color']
        tmp = df.loc[df['year'] == cur_year, cols]
        # plot_map_europe(tmp, cmap, bound, title=f"{cur_year}")

    # ==========
    # Total Year
    # ==========
    grouped_df = df.groupby('NAME')['friendly_index_net'].sum().reset_index()

    grouped_df['color'], cmap, bound = generate_color_col(
        srs=grouped_df['friendly_index_net'],
        step=3
    )
    plot_map_europe(grouped_df, cmap, bound, title="2017-2023")

    return


def main():
    task_europe_friendliness()


if __name__ == "__main__":
    main()
