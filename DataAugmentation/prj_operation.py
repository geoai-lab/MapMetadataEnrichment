import geopandas as gpd
import matplotlib
import os
import matplotlib.pyplot as plt
from DataAugmentation.dictionary_proj4 import State_proj4, Continent_proj4
matplotlib.use('TkAgg')


root_dir_world = "../Training_data/shapefiles/Continents"
root_dir_us = "../Training_data/shapefiles/States"
#result_dir_world = "../Training_data/base_image/Continents"
result_dir_us = "../Training_data/base_image/States"
result_dir_world = "../Training_data/no_augmentation/Continents"
#result_dir_us = "../Training_data/no_augmentation/States"

mycolor = (58/255,118/255,175/255)


def list_files(folder_dir):
    filenames = os.listdir(folder_dir)  # get all files' and folders' names in the current directory

    result = []
    for filename in filenames:  # loop through all the files and folders
        if os.path.isdir(os.path.join(folder_dir, filename)):  # check whether the current object is a folder or not
            result.append(filename)

    result.sort()

    return result


def re_projection(p_shpfile, p_prj):
    p_shpfile2 = p_shpfile.to_crs(p_prj)
    return p_shpfile2


def continent_process(shapefile, img_path, name, no_augmentation):
    index = 0
    for prj in Continent_proj4[name]:
        index += 1
        temp_shp = re_projection(shapefile, prj)
        print(temp_shp.crs)
        temp_shp.plot(linewidth=0.1, edgecolor = mycolor, color = mycolor)
        plt.axis('off')
        plt.savefig(img_path + '/' + str(index) + ".png", dpi=96, bbox_inches='tight', pad_inches=0)
        plt.cla()
        plt.clf()
        plt.close()
        if no_augmentation:
            break


def state_process(shapefile, img_path, name, no_augmentation):
    index = 0
    for prj in (State_proj4[name] + State_proj4['U.S.']):
        index += 1
        temp_shp = re_projection(shapefile, prj)
        print(temp_shp.crs)
        temp_shp.plot(linewidth=0.1, edgecolor = mycolor, color = mycolor)
        plt.axis('off')
        plt.savefig(img_path + '/' + str(index) + ".png", dpi=96, bbox_inches='tight', pad_inches=0)
        plt.cla()
        plt.clf()
        plt.close()
        if no_augmentation:
            break


if __name__ == "__main__":
    ## World continents reproject images generation 
#     p_list = [item for item in list_files(root_dir_world)]
#     for item in p_list:
#         print("-------- working on the {0} continent --------".format(item))
#         shapefiles = [file for file in os.listdir(os.path.join(root_dir_world, item)) if file.endswith(".shp")]
#         pshp_path = os.path.join(root_dir_world, item, shapefiles[0])
#         ppng_path = os.path.join(result_dir_world, item)
#         isExists = os.path.exists(ppng_path)
#         if not isExists:
#             os.makedirs(ppng_path)
#         shp = gpd.read_file(pshp_path)
#         continent_process(shp, ppng_path, str(shapefiles[0].split('.')[0]), no_augmentation=True)
        

    ## U.S. States reproject images generation
    p_list = [item for item in list_files(root_dir_us)]
    for item in p_list:
        print("-------- working on the {0} state --------".format(item))
        shapefiles = [file for file in os.listdir(os.path.join(root_dir_us, item)) if file.endswith(".shp")]
        pshp_path = os.path.join(root_dir_us, item, shapefiles[0])
        ppng_path = os.path.join(result_dir_us, item)
        isExists = os.path.exists(ppng_path)
        if not isExists:
            os.makedirs(ppng_path)
        shp = gpd.read_file(pshp_path)
        state_process(shp, ppng_path, str(shapefiles[0].split('.')[0]), no_augmentation=False)

