import os
import numpy as np
from osgeo import gdal
import math
import pandas as pd

path1 = 'Input/Sichuan2010_xq_Clip.tif'
path2 = 'Input/Sichuan2020_clip.tif'

path1_result = 'Sichuan_2010_v2.asc'
path2_result = 'Sichuan_2020.asc'

window_size = 33

def ReadGeoTIF(filepath):
    """ 该函数用于读取GEOTIF数据"""
    #print('开始读取数据')
    AData = gdal.Open(os.path.join(os.path.dirname(__file__),filepath))
    # 读取仿射变换函数
    data_geotrans = AData.GetGeoTransform()
    #print(data_geotrans)
    # 读取投影信息
    #data_proj = AData.GetProjection()
    # 获取栅格矩阵的列数
    data_col = AData.RasterXSize
    # 获取栅格矩阵的行数
    data_row = AData.RasterYSize
    # 获取栅格矩阵的波段数
    #data_bands = AData.RasterCount
    # 将单波段数据读取为一个numpy矩阵
    data_tif = AData.ReadAsArray(0, 0, data_col, data_row)

    # 显示栅格数据
    # ShowMap(data_tif,-9999)
    # 获取第1个波段， 需要注意的是， GDAL库中的波段是从第一个开始获取的
    band = AData.GetRasterBand(1)
    data_nodata = band.GetNoDataValue()
    #data_tif[data_tif == 0] = data_nodata  # 将0更改为nodata
    data_tif[data_tif == 255] = data_nodata  # 将0更改为nodata

    return [data_tif, data_nodata, data_geotrans]

def WriteResult(data, xllcorner,yllcorner,cellsize,count_y, ws,name):
    """
        这个函数的功能是用来输出模拟得到的结果的
    """
    size = np.shape(data)

    # 设置头文件
    head = ['','','','','','']
    head[0] = 'ncols'+ '\t' + str(size[1])
    head[1] = 'nrows' + '\t' + str(size[0])
    head[2] = 'xllcorner' + '\t' + str(xllcorner)
    head[3] = 'yllcorner' + '\t' + str(yllcorner - cellsize * count_y)
    head[4] = 'cellsize' + '\t' + str(cellsize * ws)
    head[5] = 'NODATA_value' + '\t' + str(-9999)
    
    size = np.shape(data)
    fig = open(os.path.join(os.path.dirname(__file__),'Result', name), 'w')
    # 输入头文件
    for h in range(len(head)):
        fig.write(head[h])
        fig.write("\n")

    for i in range(0,size[0]):
        for j in range(0,size[1]):
            fig.write(("%d" % data[i][j]))
            fig.write('\t')
        fig.write("\n")

def LandSystem(filepath, ws):

    "establish the land system and save the land system as the "
    """
    filepath: the path of the input raster land cover data
    ws: the size of the sliding window
    name: the name of output data
    """

    [data, nodata, data_geotrans] = ReadGeoTIF(filepath)

    size = np.shape(data)

    # the row and col of the result
    nr = math.floor(size[0] / ws)
    nc = math.floor(size[1] / ws)

    # define the land system data
    LS = np.zeros(shape=(nr,nc))

    for i in range( size[0] - 1, size[0] - (nr) * ws, -ws ):
        for j in range( 0, (nc) * ws + 1, ws ):

            # get the data of sliding window
            temp = data[i - ws + 1:i + 1, j:j + ws] * 1
            
            # new index
            size_temp = np.shape(temp)
            #print(size_temp)

            if size_temp[0] == ws & size_temp[1] == ws:
                ii = math.floor((i + 1) / ws) - 1
                jj = math.floor((j + 1) / ws)

                # calculate the porportion of each land cover type
                #cp = np.zeros(shape = landcovertype + 1)
                cp = []
                cp.append(np.sum(temp == 10)) # 10
                cp.append(np.sum(temp == 20)) # 20
                cp.append(np.sum(temp == 30)) # 30
                cp.append(np.sum(temp == 40)) # 40
                cp.append(np.sum(temp == 50)) # 50
                cp.append(np.sum(temp == 60)) # 60
                cp.append(np.sum(temp == 70) + np.sum(temp == 71) + np.sum(temp == 72) + np.sum(temp == 73) + np.sum(temp == 74)) # 70
                cp.append(np.sum(temp == 80)) # 80
                cp.append(np.sum(temp == 90)) # 90
                cp.append(np.sum(temp == 100)) # 100
                cp.append(np.sum(temp == nodata)) # 7

                #max_index = cp.index(max(cp))

                
                b = sorted(zip(cp, range(len(cp))))
                b.sort(key = lambda x : x[0], reverse = True)  # x[0]是因为在元组中，按a排序，a在第0位,这里的x不是前面的数组x，只是临时申请的变量
                c = [x[1] for x in b]  # x[1]是因为在元组中，下标在第1位
                
                max_index = c[0]
                second_index = c[1]


                if max_index == 10:

                    # 最后一个是nodata
                    LS[ii,jj] = -9999

                else:
                    
                    density = cp[max_index] / (ws * ws)

                    # crop system

                    if max_index == 0:
                        if density > 0.8:
                            LS[ii,jj] = 0 # dense crop
                        else:
                            if second_index == 1:
                                LS[ii,jj] = 1 # crop & forest mosaic
                            else:
                                if second_index == 2:
                                    LS[ii,jj] = 2 # crop & grass mosaic
                                else:
                                    LS[ii,jj] = 3 # spare crop

                    # forest system

                    if max_index == 1:
                        if density > 0.8:
                            LS[ii,jj] = 4 # dense forest
                        else:
                            if second_index == 0:
                                LS[ii,jj] = 5 # forest & crop mosaic
                            else:
                                if second_index == 2:
                                    LS[ii,jj] = 6 # forest & grass mosaic
                                else:
                                    LS[ii,jj] = 7 # spare forest

                    # grass system

                    if max_index == 2:
                        if density > 0.8:
                            LS[ii,jj] = 9 # dense grass
                        else:
                            if second_index == 1:
                                LS[ii,jj] = 10 # grass & forest mosaic
                            else:
                                if second_index == 4:
                                    LS[ii,jj] = 11 # grass & wetland mosaic
                                else:
                                    LS[ii,jj] = 12 # spare grass

                    # shrub system

                    if max_index == 3:
                        if density > 0.8:
                            LS[ii,jj] = 13 # dense shrub
                        else:
                            if second_index == 1:
                                LS[ii,jj] = 14 # shrub & forest mosaic
                            else:
                                if second_index == 2:
                                    LS[ii,jj] = 15 # shrub & grass mosaic
                                else:
                                    LS[ii,jj] = 16 # spare shrub

                    # wetland system

                    if max_index == 4:
                        if density > 0.8:
                            LS[ii,jj] = 17 # dense wetland
                        else:
                            if second_index == 1:
                                LS[ii,jj] = 18 # wetland & forest mosaic
                            else:
                                if second_index == 2:
                                    LS[ii,jj] = 19 # wetland & grass mosaic
                                else:
                                    LS[ii,jj] = 20 # spare wetland

                    # water system

                    if max_index == 5:
                        if density > 0.8:
                            LS[ii,jj] = 21 # dense water
                        else:
                            LS[ii,jj] = 22 # spare water
                    
                    # tundra system

                    if max_index == 6:
                        if density > 0.8:
                            LS[ii,jj] = 23 # dense tundra
                        else:
                            LS[ii,jj] = 24 # spare tundra

                    # artificial surface system

                    if max_index == 7:
                        if density > 0.8:
                            LS[ii,jj] = 25 # dense artificial surfaces modaic
                        else:
                            if second_index == 0:
                                LS[ii,jj] == 26 # artificial surfaces & crop mosaic
                            else:
                                LS[ii,jj] = 27 # spare artificial surfaces mosaic

                    # bare system

                    if max_index == 8:
                        if density > 0.8:
                            LS[ii,jj] = 28 # dense bare mosaic
                        else:
                            LS[ii,jj] = 29 # spare bare mosaic

                    # ice and permanent snow system

                    if max_index == 9:
                        if density > 0.8:
                            LS[ii,jj] = 30 # dense ice and permanent snow
                        else:
                            LS[ii,jj] = 31 # spare ice and permanent snow

    # generate the land system data
                                

    return [LS, data_geotrans, size[0]]

def GetDemand():
    """
    demand
    """
    #built_up = 1630 # km2
    #crop_production = 31828400 # ton
    #forest_stock = 168000.00 # 10000 m3

    crop = 116848.2645
    forest = 196017.498
    grass = 149378.3541
    shrub = 9720.3843
    urban = 2414.8845

    return [crop, forest, grass, shrub, urban]    

def CalCapacity_evaluation(Data30_path, Data990, ws, demand, origin_resolution, result_resolution):
    
    """
    计算供给能力
    W是一个字符串
    Year是一个字符串
    Data30 is the filepath of the land cover data
    Data990 is a land system data
    ws is the size of the sliding window
    demand is the matrix of the demand
    """

    # 读取30米分辨率的GlobeLand30数据
    [Data30, Data30_nodata, data_geotrans]= ReadGeoTIF(Data30_path)

    # 确定990米分辨率的数据中有几种土地系统类型
    size = np.shape(Data990)

    ls_list_temp = [] # 用来存储可能的土地系统类型
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            if Data990[i][j] != -9999:
                ls_list_temp.append(Data990[i][j])
                #for k in ls_list:
                #    if k != Data990[i][j]:
                #        ls_list.append(Data990[i][j])

    
    # 去除重复元素
    # ls_list = list(set(ls_list_temp))
    ls_list = range(0,32)
    ls_number = len(ls_list) # 土地系统类型的数量
    #print(ls_list)
    # 将土地数据从小到大排列
    ls_list_sort = sorted(ls_list)
    #print(ls_list_sort)
    # 创建一个数组0-len（ls_num)
    #list0_len = range(0, ls_number)
    #for i in range(0, size[0]):
        #for j in range(0, size[1]):
            #if Data990[i][j] != -9999:
                #Data990[i][j] = list0_len[ls_list_sort.index(Data990[i][j])]
    # print(Data990)

    Demand = demand

    # 计算30米分辨率数据的大小
    size1 = np.shape(Data30)

    #print('30米分辨率的数据大小为：', size1, size)

    # 创建标记矩阵
    Sign = np.zeros((ls_number, size[0], size[1]))
    Num = np.zeros((5, size[0], size[1]))

    for i in np.arange(size1[0]-1,  size1[0] - (size[0]) * ws-1, -ws):
        for j in np.arange(0, (size[1]) * ws + 1, ws):

            # 获取模板数据
            temp = Data30[i - ws + 1:i + 1, j:j + ws]
            size_temp = np.shape(temp)
            #temp = Data30[i - ws + 1:i, j:j + ws - 1].copy()
            if size_temp[0] == ws & size_temp[1] == ws:

                #print(temp)
                ii = (int((i+1)/ws) - 1)
                jj = (int((j+1)/ws))

                # print(i, j, ii, jj)
                # print(ii, jj)
                # 统计滑动窗口内的耕地、林地、草地、灌木的GlobeLand30像素的个数

                crop30 = np.sum(temp == 10)
                forest30 = np.sum(temp == 20)
                grass30 = np.sum(temp == 30)
                shrub30 = np.sum(temp == 40)
                urban30 = np.sum(temp == 80)

                Num[0, ii, jj] = crop30
                Num[1, ii, jj] = forest30
                Num[2, ii, jj] = grass30
                Num[3, ii, jj] = shrub30
                Num[4, ii, jj] = urban30

            
            if Data990[ii, jj] != -9999:
                Sign[int(Data990[ii, jj]), ii, jj] = 1

    #print(Num)
    #print('开始计算供给能力！')
    Sum_Crop = np.zeros((ls_number))
    Sum_Forest = np.zeros((ls_number))
    Sum_Grass = np.zeros((ls_number))
    Sum_Shrub = np.zeros((ls_number))
    Sum_Urban = np.zeros((ls_number))

    Area = np.zeros((ls_number))
    for t in range(0,ls_number):
        # 在Python中，实现对应元素相乘，有2种方式，一个是np.multiply()，另外一个是*。见如下Python代码：
        Sum_Crop[t] = np.sum(np.multiply(Sign[t, :, :], Num[0, :, :]))
        Sum_Forest[t] = np.sum(np.multiply(Sign[t, :, :], Num[1, :, :]))
        Sum_Grass[t] = np.sum(np.multiply(Sign[t, :, :], Num[2, :, :]))
        Sum_Shrub[t] = np.sum(np.multiply(Sign[t, :, :], Num[3, :, :]))
        Sum_Urban[t] = np.sum(np.multiply(Sign[t, :, :], Num[4, :, :]))

        # 此时的单位是 km2
        Area[t] = np.sum(Sign[t, :, :]) * (origin_resolution * ws) * (origin_resolution * ws) / 1000000
    #print(Num[0,:,:],Sum_Crop)
    if np.sum(Sum_Crop) == 0:
        crop_d = np.zeros((ls_number, 1))
    else:
        crop_d = Sum_Crop / np.sum(Sum_Crop)

    if np.sum(Sum_Forest) == 0:
        forest_d = np.zeros((ls_number, 1))
    else:
        forest_d = Sum_Forest / np.sum(Sum_Forest)

    if np.sum(Sum_Grass) == 0:
        grass_d = np.zeros((ls_number, 1))
    else:
        grass_d = Sum_Grass / np.sum(Sum_Grass)
    
    if np.sum(Sum_Shrub) == 0:
        shrub_d = np.zeros((ls_number, 1))
    else:
        shrub_d = Sum_Shrub / np.sum(Sum_Shrub)

    if np.sum(Sum_Urban) == 0:
        urban_d = np.zeros((ls_number, 1))
    else:
        urban_d = Sum_Urban / np.sum(Sum_Urban)

    # 计算供给能力
    Capacity = np.zeros((ls_number, len(demand)))

    for t in range(0, ls_number):

        if Area[t] == 0:
            Capacity[t, 0] = 0
            Capacity[t, 1] = 0
            Capacity[t, 2] = 0
            Capacity[t, 3] = 0
            Capacity[t, 4] = 0

        else:
            Capacity[t, 0] = ( Demand[0] * crop_d[t] ) / Area[t] / (result_resolution * result_resolution / 1000000 ) 
            Capacity[t, 1] = ( Demand[1] * forest_d[t] ) / Area[t] / (result_resolution * result_resolution / 1000000 ) 
            Capacity[t, 2] = ( Demand[2] * grass_d[t] ) / Area[t] / (result_resolution * result_resolution / 1000000 ) 
            Capacity[t, 3] = ( Demand[3] * shrub_d[t] ) / Area[t] / (result_resolution * result_resolution / 1000000 ) 
            Capacity[t, 4] = ( Demand[4] * urban_d[t] ) / Area[t] / (result_resolution * result_resolution / 1000000 ) 

    # 判断这个流域需求的数量
    demand_count = 0
    for d in range(0, len(demand)):
        if np.sum(Capacity[:, d]) != 0:
            demand_count = demand_count + 1

    # 新建一个矩阵存储计算结果
    Capacity_final = np.zeros((ls_number, demand_count))
    #print(Capacity)
    # 整理计算结果
    dd = 0
    for d in range(0,len(demand)):
        #print(Capacity_final[:][d])
        if np.sum(Capacity[:, d]) != 0:
            for l in range(0, ls_number):
                Capacity_final[l, dd] = Capacity[l, d]
            dd = dd + 1

    size_capacity_2 = np.shape(Capacity_final)
    fig = open(os.path.join(os.path.dirname(__file__),'Result/lusmatrix.txt'), 'w')
    for n in range(0, size_capacity_2[0]):
        for m in range(0, size_capacity_2[1]):
            fig.write(("%.4f" % Capacity_final[n][m]))
            fig.write('\t')
        fig.write('\n')

    # 关闭文件流
    fig.close()

    return Capacity_final


[LS1_result, geo, count] = LandSystem(path1, 33)
[LS2_result, geo, count] = LandSystem(path2, 33)

WriteResult(LS1_result, geo[0], geo[3], geo[1], count, window_size, path1_result)
WriteResult(LS2_result, geo[0], geo[3], geo[1], count, window_size, path2_result)

demand_result = GetDemand()

CalCapacity_evaluation(path1, LS1_result, window_size, demand_result, 30, 990)
