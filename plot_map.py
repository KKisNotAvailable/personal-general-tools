
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
'''
https://www.naturalearthdata.com/downloads/110m-cultural-vectors/

https://jan-46106.medium.com/plotting-maps-with-european-data-in-python-part-i-decd83837de4
'''


# load the low resolution world map
def plot_map(area: str):
    countries = gpd.read_file('./data/ne_110m_admin_0_countries/')
    countries.head(5)


def main():
    plot_map("europe")


if __name__ == "__main__":
    main()