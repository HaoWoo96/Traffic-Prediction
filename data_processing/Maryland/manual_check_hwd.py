import pandas as pd
import pickle
import networkx as nx
import numpy as np
import geopandas as gpd
import json

from shapely.geometry import  MultiLineString, mapping, shape
from matplotlib import pyplot as plt

# TMC & XD Geometry Data
gdf_hwd_tmc = gpd.read_file("/Volumes/Hao's WD/Howard/Shapefiles/tmc_shape_hwd.geojson")  # 1093 TMC segments in Howard, MD 
gdf_hwd_tmc = gdf_hwd_tmc.rename(columns={"tmc":"id_tmc", "miles":"miles_tmc"})
gdf_hwd_tmc.miles_tmc = gdf_hwd_tmc.miles_tmc.astype(float)

gdf_hwd_xd = gpd.read_file("/Volumes/Hao's WD/Howard/Shapefiles/xd_shape_hwd_for_sjoin.geojson")  # 1772 of 4723 XD segments in Howard, MD; used for spatial join
gdf_hwd_xd = gdf_hwd_xd.rename(columns={"xd":"id_xd_str", "miles":"miles_xd"})
gdf_hwd_xd.id_xd_str = gdf_hwd_xd.id_xd_str.astype(str)
gdf_hwd_xd.miles_xd = gdf_hwd_xd.miles_xd.astype(float)


'''
For manual check of spatially joined TMC & XD segments
'''
raw_xd_to_tmc = pd.read_csv("./hwd_raw_xd_to_tmc.csv")
raw_xd_to_tmc.id_xd = raw_xd_to_tmc.id_xd.astype(str)
raw_xd_to_tmc = raw_xd_to_tmc.sort_values(by="angle", ascending=False)

# Visualization for Checking
for i in range(563, raw_xd_to_tmc.shape[0]):
    fig, ax = plt.subplots(1,figsize=(7.5,7.5), dpi=100)
    curr_tmc = raw_xd_to_tmc.iloc[i].id_tmc
    curr_xd = raw_xd_to_tmc.iloc[i].id_xd
    # gdf_tmc_cranberry.plot(ax=ax, color="grey", alpha=0.1)

    plt.title(f"index = {i+2} \n \
            curr_tmc (blue): {curr_tmc}    direction: {raw_xd_to_tmc.iloc[i].direction_tmc}     ratio_tmc: {raw_xd_to_tmc.iloc[i].overlap_vs_tmc} \n \
            curr_xd (red): {curr_xd}      direction: {raw_xd_to_tmc.iloc[i].direction_xd}       ratio_xd: {raw_xd_to_tmc.iloc[i].overlap_vs_xd}", \
                fontsize=7)
    gdf_hwd_tmc[gdf_hwd_tmc.id_tmc == curr_tmc].plot(ax=ax, color="Blue", alpha=0.3)
    gdf_hwd_xd[gdf_hwd_xd.id_xd_str == curr_xd].plot(ax=ax, color="Red", alpha=0.5)
    plt.show()


'''
For manual check of connected TMC neighbors
'''
# df_raw_prev_tmc = pd.read_csv("./raw_prev_tmc_v2.csv")
# df_raw_prev_tmc = df_raw_prev_tmc.sort_values(by="angle", ascending=False)

# # Visualization for Checking
# for i in range(df_raw_prev_tmc.shape[0]):
#     check_curr_tmc = df_raw_prev_tmc.iloc[i].tmc_x
#     check_prev_tmc = df_raw_prev_tmc.iloc[i].tmc_y
#     # print(check_curr_tmc, df_raw_prev_tmc.iloc[i].direction_x,check_prev_tmc, df_raw_prev_tmc.iloc[i].direction_y, df_raw_prev_tmc.iloc[i].angle)
#     fig, ax = plt.subplots(1,figsize=(7.5,6), dpi=100)
#     # plt.title(f"index = {i+2} \n \
#     #         angle = {df_raw_prev_tmc.iloc[i].angle}\n \
#     #         (RED) curr_tmc: {check_curr_tmc}    direction: {df_raw_prev_tmc.iloc[i].direction_x}    name: {df_raw_prev_tmc.iloc[i].roadname_x}      start: {df_raw_prev_tmc.iloc[i].start_latitude_x}, {df_raw_prev_tmc.iloc[i].start_longitude_x}     end: {df_raw_prev_tmc.iloc[i].end_latitude_x}, {df_raw_prev_tmc.iloc[i].end_longitude_x} \n \
#     #         (BLUE) prev_tmc: {check_prev_tmc}      direction: {df_raw_prev_tmc.iloc[i].direction_y}     name: {df_raw_prev_tmc.iloc[i].roadname_x}      start: {df_raw_prev_tmc.iloc[i].start_latitude_y}, {df_raw_prev_tmc.iloc[i].start_longitude_y}      end: {df_raw_prev_tmc.iloc[i].end_latitude_y}, {df_raw_prev_tmc.iloc[i].end_longitude_y} ", \
#     #             fontsize=7)
#     plt.title(f"index = {i+2} \n \
#         angle = {df_raw_prev_tmc.iloc[i].angle}\n \
#         (RED) curr_tmc: {check_curr_tmc}    direction: {df_raw_prev_tmc.iloc[i].direction_x}    name: {df_raw_prev_tmc.iloc[i].roadname_x} \n \
#         (BLUE) prev_tmc: {check_prev_tmc}      direction: {df_raw_prev_tmc.iloc[i].direction_y}     name: {df_raw_prev_tmc.iloc[i].roadname_y}", \
#             fontsize=7)
#     # gdf_tmc_cranberry.plot(ax=ax, color="grey", alpha=0.3)
#     gdf_tmc_cranberry[gdf_tmc_cranberry.tmc == check_curr_tmc].plot(ax=ax, color="red", alpha=0.5)
#     gdf_tmc_cranberry[gdf_tmc_cranberry.tmc == check_prev_tmc].plot(ax=ax, color="blue", alpha=0.3)
#     plt.show()


'''
For manual verification of computed upstream segments
'''

# # load dictionary of upstream
# with open("../../data/dict_upstream_tmc_v2.pkl", "rb") as f:
#     dict_upstream_tmc = pickle.load(f)

# for check_curr_tmc in dict_upstream_tmc:
#     print("current tmc: ", check_curr_tmc, gdf_tmc_cranberry[gdf_tmc_cranberry.id_tmc == check_curr_tmc].direction)
#     check_upstream = dict_upstream_tmc[check_curr_tmc]
#     print("upstream: ", check_upstream)
#     fig, ax = plt.subplots(1,figsize=(10,7.5), dpi=120)
#     gdf_tmc_cranberry.plot(ax=ax, color="grey", alpha=0.05) # entire TMC segments in cranberry 
#     gdf_tmc_cranberry[gdf_tmc_cranberry.id_tmc == check_curr_tmc].plot(ax=ax, color="orange") # source
#     if len(check_upstream) > 1:
#         print(f"MORE THAN ONE UPSTREAM PATH!!! There are {len(check_upstream)} upstream paths")
#     j = 0
#     for k, u in enumerate(check_upstream[j]):
#         gdf_tmc_cranberry[gdf_tmc_cranberry.id_tmc == u].plot(ax=ax, color="blue", alpha= (len(check_upstream[j])-k)/len(check_upstream[j]))  # upstream segments
#     plt.show()


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