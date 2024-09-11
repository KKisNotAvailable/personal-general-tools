import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import geopandas as gpd
from PIL import Image
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")
'''
Map geometry source:
https://www.naturalearthdata.com/downloads/110m-cultural-vectors/
'''

OUTPUT_PATH = './output/'
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# TODO: 
# 1. remove redundant file reading part
# 2. make the arguments kwargs (and need_colorbar=T/F)

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
        title: str = None,
        fig=None,
        ax=None,
        is_sub=False
    ):
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

    # 1. get data of other countries
    # Include Cyprus (for current need) and exclude Russia (too large)
    countries = gpd.read_file('./data/ne_10m_admin_0_countries/')  # high res

    cond = (countries['NAME'] != "Russia") &\
        (countries['CONTINENT'] == "Europe") |\
        (countries['NAME'] == "Cyprus")

    countries = countries.loc[cond, cols_to_keep].reset_index(drop=True)

    # 2. combine with our data (with color column in it)
    countries = pd.merge(countries, extra_data, on='NAME', how='left')

    # if country no color assigned, set it to be gray 
    # (but not supposed to have for current setup)
    countries.fillna({'color': '#808080'}, inplace=True)

    ax.set_axis_off() # turn the axis off
    countries.plot(
        ax=ax, 
        color=countries.color,
        edgecolor='black',        # Color of the edges
        linewidth=0.5             # Width of the edges
    )

    # setting the display limit solves the territory problem
    ax.set_xlim([-25, 45])  # Longitude
    ax.set_ylim([30, 72])   # Latitude

    ax.set_aspect(1.5)  # adjust the display ratio

    # plt.gcf().set_facecolor('dodgerblue') # set background color

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
    
    cax = ax.inset_axes([0.1, 0.1, 0.02, 0.3], transform=ax.transAxes) # set the colorbar position and size
    cbar = fig.colorbar(
        sm, cax=cax, fraction=0.035, pad=0.04,
        orientation='vertical'
    )
    np_bound = np.array(bound)
    normalized = (np_bound - min(np_bound)) / (max(np_bound) - min(np_bound))
    cbar.set_ticks(normalized)
    cbar.set_ticklabels(
        np.round(bound, 0).astype(int), 
        ha='left', # horizontal alignment
        # x=2.0 # the starting point of label, =0 will be overlapping with the colorbar
    )
    cbar.ax.tick_params(labelsize=6)

    cbar.set_label('Friendly Index', fontsize=8)
    cbar.ax.yaxis.set_label_position('left')
    # cbar.ax.set_position([0.17, 0.15, 0.03, 0.38])

    if title:
        ax.set_title(title, fontsize=16, weight='bold')

    # if is not subplot
    if not is_sub:
        # make the margin smaller
        plt.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.05)
        
        # plt.savefig(OUTPUT_PATH + f'{title}.svg')
        plt.show()
        plt.close()

    return


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
    # 1. Prepare the friendliness data
    df = pd.read_stata("./data/friendly_index_net.dta")
    cols_to_keep = ['country', 'year', 'friendly_index_net']
    df = df[cols_to_keep]
    df['year'] = df['year'].astype('int32')

    # add UK and set to be 0 (to prevent it being gray)
    UK_data = pd.DataFrame({
        'country': ["英國"] * 7,
        'year': [n for n in range(2017, 2024)],
        'friendly_index_net': [0.0] * 7
    })

    df = pd.concat([df, UK_data]).reset_index(drop=True)

    # 2. Mapping the chinese names in friendliness data to english
    name_map = pd.read_csv("./data/eu_country_name_mapping.csv")

    df = pd.merge(df, name_map, on='country', how='left')

    # 3. For yearly data, 
    #    - prepare color column 
    #    - set the color map
    #    - set the friendliness index range (for color bar display)
    df['color'], cmap, bound = generate_color_col(
        srs=df['friendly_index_net'],
        step=3
    )

    # 4. Plot the yearly data
    y_list = sorted(set(df['year']))
    sizex, sizey = 12, 9
    cols = ['NAME', 'friendly_index_net', 'color']

    for cur_year in y_list:
        # not all countries are included in each yaer
        tmp = df.loc[df['year'] == cur_year, cols]
        fig, axs = plt.subplots(1, 1, figsize=(sizex, sizey))
        plot_map_europe(tmp, cmap, bound, title=f"{cur_year}", ax=axs, fig=fig)
    
    # 5. Plot the aggregate years
    #    - get combined data
    #    - prepare color things
    grouped_df = df.groupby('NAME')['friendly_index_net'].sum().reset_index()

    grouped_df['color'], cmap_all, bound_all = generate_color_col(
        srs=grouped_df['friendly_index_net'],
        step=3
    )
    fig, axs = plt.subplots(1, 1, figsize=(sizex, sizey))
    plot_map_europe(grouped_df, cmap_all, bound_all, title="2017-2023", ax=axs, fig=fig)
    
    # 6. Plot aggr, 2017, 2020, 2023 on the same image
    #    - prepare frames for subplots
    #    - plot each designated year again onto subplots
    fig, axs = plt.subplots(2, 2, figsize=(sizex, sizey))

    # top-left
    cur_axs = axs[0, 0]
    cur_bound = [min(bound_all), 0, max(bound_all)]
    plot_map_europe(
        grouped_df, cmap_all, cur_bound, title="2017-2023", 
        ax=cur_axs, fig=fig, is_sub=True
    )

    cur_bound = [min(bound), 0, max(bound)]
    # top-right -> bottom-left -> bottom-right
    for cur_year, cur_axs in zip([2017, 2020, 2023], [axs[0, 1], axs[1, 0], axs[1, 1]]):
        tmp = df.loc[df['year'] == cur_year, cols] # "cols" is the same when plotting yearly
        plot_map_europe(
            tmp, cmap, cur_bound, title=f"{cur_year}", 
            ax=cur_axs, fig=fig, is_sub=True
        )

    plt.subplots_adjust(wspace=-0.3, hspace=0.1) # Adjust horizontal and vertical space
    # plt.savefig(OUTPUT_PATH + 'combined.svg')
    plt.show()

    return


def main():
    task_europe_friendliness()


if __name__ == "__main__":
    main()
