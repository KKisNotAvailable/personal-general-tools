import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import geopandas as gpd
from PIL import Image
'''
https://www.naturalearthdata.com/downloads/110m-cultural-vectors/

https://jan-46106.medium.com/plotting-maps-with-european-data-in-python-part-i-decd83837de4
'''


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


def plot_map_europe(extra_data: pd.DataFrame = None, cmap=None):
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

    cond = (countries['CONTINENT'] == "Europe") |\
        (countries['NAME'] == "Cyprus")

    countries = countries.loc[cond, cols_to_keep].reset_index(drop=True)
    countries = pd.merge(countries, extra_data, on='NAME', how='left')

    # countries['color'].fillna('#808080', inplace=True) # will not work in future
    countries.fillna({'color': '#808080'}, inplace=True)

    sizex, sizey = 12, 9
    ax = countries.plot(figsize=(sizex, sizey), color=countries.color)

    # actually, setting the display limit solves the territory problem
    ax.set_xlim([-25, 45])  # Longitude
    ax.set_ylim([30, 72])   # Latitude

    ax.set_aspect(1.5)  # adjust the display ratio

    sm = plt.cm.ScalarMappable(cmap=cmap)
    cbar = plt.colorbar(
        sm, ax=ax, fraction=0.035, pad=0.04,
        orientation='horizontal'
    )
    ticks = [
        'Very High', 'High', 'Neutral High',
        'Neutral Low', 'Low', 'Very Low'
    ]
    cbar.set_ticklabels(ticks)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('Friendly Index')

    plt.axis("off")
    plt.show()


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
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_color", colors, N=levels)

    norm = mcolors.Normalize(vmin=0, vmax=100)
    return [mcolors.to_hex(cmap(norm(p))) for p in srs], cmap


def task_europe_friendliness():
    '''
    ['country', 'year', 'unfriendly_cnt', 'friendly_cnt',
    'friendly_index_net']
    '''
    df = pd.read_stata("./data/friendly_index_net.dta")
    cols_to_keep = ['country', 'year', 'friendly_index_net']
    df = df[cols_to_keep]
    df['year'] = df['year'].astype('int32')

    # df.to_csv(
    #     "./data/eu_country_name_mapping.csv",
    #     index=False, encoding='utf-8-sig'
    # )
    # return

    name_map = pd.read_csv("./data/eu_country_name_mapping.csv")

    df = pd.merge(df, name_map, on='country', how='left')

    df['color'], cmap = generate_color_col(
        srs=df['friendly_index_net'].rank(pct=True) * 100,
        levels=10
    )

    cur_year = 2017

    # not all countries are included in each yaer
    df = df.loc[df['year'] == cur_year, ['NAME', 'color']]
    print(df.columns)
    plot_map_europe(df, cmap)
    # 全部加總的顏色要另外等加總完之後再做

    return


def main():
    task_europe_friendliness()
    # plot_map_europe()
    # plot_map_world(area="Europe")


if __name__ == "__main__":
    main()
