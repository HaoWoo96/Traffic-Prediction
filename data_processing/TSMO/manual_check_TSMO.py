import pandas as pd
import pickle
import networkx as nx
import numpy as np
import geopandas as gpd
import json

from shapely.geometry import  MultiLineString, mapping, shape
from matplotlib import pyplot as plt

# TMC & XD Geometry Data
gdf_TSMO_tmc = gpd.read_file("./data/shape/tmc_shape_TSMO/tmc_shape_TSMO_for_sjoin.geojson")  # 1591 TMC segments in TSMO used for spatial join
gdf_TSMO_tmc = gdf_TSMO_tmc.rename(columns={"tmc":"id_tmc", "miles":"miles_tmc"})
gdf_TSMO_tmc.miles_tmc = gdf_TSMO_tmc.miles_tmc.astype(float)

gdf_TSMO_xd = gpd.read_file("./data/shape/xd_shape_TSMO/xd_shape_TSMO_for_sjoin.geojson")  # 2501 XD segments in TSMO used for spatial join
gdf_TSMO_xd = gdf_TSMO_xd.rename(columns={"xd":"id_xd_str", "miles":"miles_xd"})
gdf_TSMO_xd.id_xd_str = gdf_TSMO_xd.id_xd_str.astype(str)
gdf_TSMO_xd.miles_xd = gdf_TSMO_xd.miles_xd.astype(float)


'''
For manual check of spatially joined TMC & XD segments
'''
# raw_xd_to_tmc = pd.read_csv("./data_processing/TSMO/raw_xd_to_tmc_TSMO.csv")
# raw_xd_to_tmc["id_xd_str"] = raw_xd_to_tmc.id_xd.astype(str)  # Be careful: cannot write "raw_xd_to_tmc.id_xd_str = raw_xd_to_tmc.id_xd.astype(str)"
# raw_xd_to_tmc = raw_xd_to_tmc.sort_values(by="angle", ascending=False)

# # Visualization for Checking
# for i in range(raw_xd_to_tmc.shape[0]):
#     fig, ax = plt.subplots(1,figsize=(7.5,7.5), dpi=200)
#     curr_tmc = raw_xd_to_tmc.iloc[i].id_tmc
#     curr_xd = raw_xd_to_tmc.iloc[i].id_xd_str
#     # gdf_tmc_cranberry.plot(ax=ax, color="grey", alpha=0.1)

#     # plt.title(f"index = {i+2} \n \
#     #         curr_tmc (blue): {curr_tmc}    direction: {raw_xd_to_tmc.iloc[i].direction_tmc}     ratio_tmc: {raw_xd_to_tmc.iloc[i].overlap_vs_tmc} \n \
#     #         curr_xd (red): {curr_xd}      direction: {raw_xd_to_tmc.iloc[i].direction_xd}       ratio_xd: {raw_xd_to_tmc.iloc[i].overlap_vs_xd}", \
#     #             fontsize=7)
#     gdf_TSMO_tmc[gdf_TSMO_tmc.id_tmc == curr_tmc].plot(ax=ax, color="Blue", alpha=0.5)
#     gdf_TSMO_xd[gdf_TSMO_xd.id_xd_str == curr_xd].plot(ax=ax, color="Red", alpha=0.5)
#     gdf_TSMO_tmc.plot(ax=ax, color="blue", alpha=0.1)
#     gdf_TSMO_xd.plot(ax=ax, color="red", alpha=0.1)
#     plt.show()


'''
For manual check of connected TMC neighbors
'''
# df_raw_prev_tmc = pd.read_csv("./data_processing/TSMO/raw_prev_tmc_TSMO.csv")
# df_raw_prev_tmc = df_raw_prev_tmc.sort_values(by="angle", ascending=False)

# # Visualization for Checking
# for i in range(1400,df_raw_prev_tmc.shape[0]):
# # check_curr_tmc = "110P53870"
# # check_prev_tmc = "110N53870"
# # i = df_raw_prev_tmc[(df_raw_prev_tmc.id_tmc_x == check_curr_tmc) & (df_raw_prev_tmc.id_tmc_y == check_prev_tmc)].index
#     check_curr_tmc = df_raw_prev_tmc.iloc[i].id_tmc_x
#     check_prev_tmc = df_raw_prev_tmc.iloc[i].id_tmc_y
#     fig, ax = plt.subplots(2, figsize=(8, 9))
#     fig.suptitle(f"index = {i+2} \n \
#         angle = {df_raw_prev_tmc.iloc[i].angle}\n \
#         (RED) curr_tmc: {check_curr_tmc}    direction: {df_raw_prev_tmc.iloc[i].direction_x}    name: {df_raw_prev_tmc.iloc[i].roadname_x} \n \
#         (BLUE) prev_tmc: {check_prev_tmc}      direction: {df_raw_prev_tmc.iloc[i].direction_y}     name: {df_raw_prev_tmc.iloc[i].roadname_y}", \
#             fontsize=7)
#     gdf_TSMO_tmc.plot(ax=ax[0], color="grey", alpha=0.1)
#     gdf_TSMO_tmc[gdf_TSMO_tmc.id_tmc == check_curr_tmc].plot(ax=ax[0], color="red", alpha=0.5)
#     gdf_TSMO_tmc[gdf_TSMO_tmc.id_tmc == check_prev_tmc].plot(ax=ax[0], color="blue", alpha=0.3)
#     gdf_TSMO_tmc[gdf_TSMO_tmc.id_tmc == check_curr_tmc].plot(ax=ax[1], color="red", alpha=0.5)
#     gdf_TSMO_tmc[gdf_TSMO_tmc.id_tmc == check_prev_tmc].plot(ax=ax[1], color="blue", alpha=0.3)
#     plt.show()


'''
For manual verification of computed upstream segments
'''

# load dictionary of upstream
with open("./data/TSMO/TSMO_dict_upstream_tmc_5_miles.pkl", "rb") as f:
    dict_upstream_tmc = pickle.load(f)

for check_curr_tmc in dict_upstream_tmc:
    print("current tmc: ", check_curr_tmc, gdf_TSMO_tmc[gdf_TSMO_tmc.id_tmc == check_curr_tmc].direction)
    check_upstream = dict_upstream_tmc[check_curr_tmc]
    if check_upstream is None:
        print(f"source {check_curr_tmc} doesn't have any upstream segments")
        continue
    else:
        check_upstream = list(check_upstream)
    print("upstream: ", check_upstream)
    fig, ax = plt.subplots(1,figsize=(10,7.5), dpi=120)
    gdf_TSMO_tmc.plot(ax=ax, color="grey", alpha=0.05) # entire TMC segments in cranberry 
    gdf_TSMO_tmc[gdf_TSMO_tmc.id_tmc == check_curr_tmc].plot(ax=ax, color="orange") # source
    if len(check_upstream) > 1:
        print(f"MORE THAN ONE UPSTREAM PATH!!! There are {len(check_upstream)} upstream paths")
    j = 0
    for k, u in enumerate(check_upstream[j]):
        gdf_TSMO_tmc[gdf_TSMO_tmc.id_tmc == u].plot(ax=ax, color="blue", alpha= (len(check_upstream[j])-k)/len(check_upstream[j]))  # upstream segments
    plt.show()


'''
For manual check of connected XD neighbors
'''
# df_raw_prev_xd_comp_checked_angle = pd.read_csv("./raw_prev_xd_new_comp_v2.csv")
# df_raw_prev_xd_comp_checked_angle = df_raw_prev_xd_comp_checked_angle.sort_values(by="angle", ascending=False)
# df_raw_prev_xd_comp_checked_angle.id_xd_str_x = df_raw_prev_xd_comp_checked_angle.id_xd_str_x.astype(str)
# df_raw_prev_xd_comp_checked_angle.id_xd_prev_str = df_raw_prev_xd_comp_checked_angle.id_xd_prev_str.astype(str)

# # Visualization for Checking
# for i in range(df_raw_prev_xd_comp_checked_angle.shape[0]):
#     check_curr_xd = df_raw_prev_xd_comp_checked_angle.iloc[i].id_xd_str_x
#     check_prev_xd = df_raw_prev_xd_comp_checked_angle.iloc[i].id_xd_prev_str
#     fig, ax = plt.subplots(1,figsize=(7.5,6), dpi=100)
#     plt.title(f"index = {i+2} \n \
#         angle = {df_raw_prev_xd_comp_checked_angle.iloc[i].angle}\n \
#         (RED) curr_xd: {check_curr_xd}    direction: {df_raw_prev_xd_comp_checked_angle.iloc[i].Bearing_x}\n \
#         (BLUE) prev_xd: {check_prev_xd}      direction: {df_raw_prev_xd_comp_checked_angle.iloc[i].Bearing_y}", \
#             fontsize=7)
#     gdf_shp_xd[gdf_shp_xd.id_xd_str == check_curr_xd].plot(ax=ax, color="red", alpha=0.5)
#     gdf_shp_xd[gdf_shp_xd.id_xd_str == check_prev_xd].plot(ax=ax, color="blue", alpha=0.3)
#     plt.show()