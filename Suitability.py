import numpy as np
from osgeo import gdal
import os

os.environ["SM_FRAMEWORK"] = "tf.keras"
import time
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.layers import Activation, Dense, Dropout
import keras.backend as K

sample1 = 'Suitability/sample1.asc'
sample2 = 'Suitability/sample2.asc'

def ReadMain(filepath,fig):
    f = open(filepath)               
    lines = f.readlines()
    GPU_device = re.split('\t| |\n',lines[0])
    reg_method = re.split('\t| |\n',lines[1])
    cell_condition = re.split('\t| |\n',lines[2])
    f.close()

    while '' in GPU_device:
        GPU_device.remove('')
    while '' in reg_method:
        reg_method.remove('')
    while '' in cell_condition:
        cell_condition.remove('')

    fig.write('The GPU is : ' + str(GPU_device) + '\n')

    if reg_method[0] == 0:
        fig.write('You select the Random Forest Regression from sklearn.' + '\n')
    
    if reg_method[0] == 1:
        fig.write('You select the ANN Regression from Tensorflow.' + '\n')

    if reg_method[0] == 2:
        fig.write('You select the Random Forest Regression from Tensorflow.' + '\n')

    if reg_method[0] == 3:
        fig.write('You select the Random Forest Regression from Tensorflow(Sample).' + '\n')

    fig.write('If the number of samples is less than ' + str(cell_condition[0]) + ', the regression will not be execute.' + '\n')

    return [int(GPU_device[0]), int(reg_method[0]), int(cell_condition[0])]


def CountDF(filepath,fig):
      
    fileList = []    
    for root, subDirs, files in os.walk(filepath):
        for fileName in files:
            if fileName.endswith('.asc'):
                if fileName[0] == 'D':
                    fileList.append(os.path.join(root,fileName))

    fig.write('The number of driving factors:' + str(len(fileList)) + '\n')

    return len(fileList)

def ReadGeoASC(filepath):
    """ Read raster data in ASC format """

    dataset = gdal.Open(os.path.join(os.path.dirname(__file__),filepath))
    # print(dataset)
    im_width = dataset.RasterXSize  # Get the row of the data

    im_height = dataset.RasterYSize  # Get the col of the data
    #print(im_width, im_height)
    band = dataset.GetRasterBand(1)  # Get the band
    im_datas = band.ReadAsArray(0, 0, im_width, im_height)  # Get the data
    
    im_geotrans = dataset.GetGeoTransform()
    """
        xllcorner = im_geotrans[0]
        yllcorner = im_geotrans[3]
        cellsize = im_geotrans[1]
    """


    return [im_datas,im_geotrans]

def Sample_change(path1, path2, count_df):
    [cov1, geo1] = ReadGeoASC(path1)
    [cov2, geo2] = ReadGeoASC(path2)
    
    type_count = np.max([np.max(cov1),np.max(cov2)])+1

    cell_count = np.sum(cov1 != -9999)
    
    size = np.shape(cov1)
    
    k = 0
    cov = np.zeros(shape = (size[0] * size[1], type_count)) 
    temp = np.zeros(shape=(1, type_count))
    temp[temp == 0] = -9999
               
    for i in range(0, size[0]):
        for j in range(0, size[1]):

            if cov1[i,j] == -9999:
                cov[k,:] = temp
                k = k + 1
            else:
                if cov1[i,j] != cov2[i,j]:
                    cov[k][int(cov2[i,j])] = 1
                    k = k + 1
                else:
                    cov[k][int(cov2[i,j])] = 0
                    k = k + 1

    cov_without_nodata = np.empty(shape=(cell_count,type_count))

    p = 0
    for i in range(0, size[0] * size[1]):
        if cov[i][0] != -9999:
            cov_without_nodata[p,0:type_count] = cov[i,0:type_count] * 1
            p = p + 1

    df = np.empty(shape=(size[0] * size[1], count_df))

    fileList = []    
    for root, subDirs, files in os.walk('Suitability'):
        for fileName in files:
            if fileName.endswith('.asc'):
                if fileName[0] == 'D':
                    fileList.append(os.path.join(root,fileName))


    for df_index in range(0, count_df):
        k = 0
        [df_data, geo]= ReadGeoASC(fileList[df_index])
        for i in range(0, size[0]):
            for j in range(0, size[1]):
                df[k, df_index] = df_data[i,j]
                k = k + 1
    df_without_nodata = np.empty(shape=(cell_count, count_df))  

    p = 0
    for i in range(0, size[0] * size[1]):
        if df[i,0] != -9999:
            df_without_nodata[p,0:count_df] = df[i,0:count_df]
            p = p + 1     

    location = []
    

    for i in range(cell_count):
        if np.sum(cov_without_nodata[i,:]) != 0:
            location.append(i)

    count_label = len(location)

    cov_label = np.empty(shape=(count_label,type_count))
    df_label = np.empty(shape=(count_label,count_df))

    for i in range(count_label):
        cov_label[i,:] = cov_without_nodata[location[i],:]
        df_label[i,:] = df_without_nodata[location[i],:]

    return [cov , cov_without_nodata , df , df_without_nodata , type_count, size, cov_label, df_label]

def RandomForest(path_head, Sample_X, Sample_Y, Pred_X,  type_count, size, fig_log, count_cell):

    size_df = np.shape(Sample_X)

    f = open(path_head,'r')
    head = []
    for i in range(6):
        head.append(f.readline().strip())
    

    for i in range(0, type_count):

        y = Sample_Y[:,i]
       
        result = []
        
        X_train, X_test, y_train, y_test = train_test_split(Sample_X,
                                                            y,
                                                            test_size = 0.75,
                                                            random_state = 0,
                                                            )

        if sum(y_train) <= count_cell:
            
            for j in range(0, size[0] * size[1]):
                
                if Pred_X[j][0] == -9999:
                    result.append(-9999)
                else:
                    result.append(0)
            
            fig_log.write('The land system ' + str(i) + ' has not enough samples to regression.' + '\n')
            fig_log.write('____________________________________________________________________' + '\n')
            
            fig_log.flush()
            fig = open(os.path.join(os.path.dirname(__file__),'Data/prob1_'+str(i)+'.1.asc'), 'w')
            for m in range(0, len(head)):
                fig.write(head[m])
                fig.write('\n')
            for n in range(0, len(result)):
                fig.write(str(result[n]))
                fig.write('\n')
            
            fig.close()
        else:
           
            model = RandomForestRegressor(n_estimators = 200, random_state = 0, oob_score = True, max_features = size_df[1],n_jobs=-1)
            
            model.fit(X_train,y_train)
            
            features_imp = model.feature_importances_ 
            

            y_test_pred = model.predict(X_test)  
            
            train_auc = roc_auc_score(y_test, y_test_pred)
            """
            sklearn.metrics.roc_auc_score(y_true, y_score, *, average='macro', sample_weight=None, max_fpr=None, multi_class='raise', labels=None)
            """
            
            fig_log.write('The AUC of the land system ' + str(i) + ' :' +  str(train_auc) + '\n')
            for imp in range(size_df[1]):
                fig_log.write('The importance of the driving factor '+ str(imp) + ' is : ' + str((features_imp[imp])) + '\n')
            fig_log.write('____________________________________________________________________' + '\n')
            fig_log.flush()

            result_pred = model.predict(Sample_X)
 
            t = 0
            for j in range(0, size[0] * size[1]):
                
                if Pred_X[j][0] == -9999:
                    result.append(-9999)
                else:
                   
                    result.append(result_pred[t])
                    t = t + 1

            fig = open(os.path.join(os.path.dirname(__file__),'Data/prob1_'+str(i)+'.1.asc'), 'w')
            for m in range(0, len(head)):
                fig.write(head[m])
                fig.write('\n')
            for n in range(0, len(result)):
                fig.write(("%.8f" % result[n]))
                fig.write('\n')
               
            fig.close()

fig_log = open(os.path.join(os.path.dirname(__file__),'Data/log_regression.fil'), 'w')
fig_log.write('Start of simulation at: ')
fig_log.write(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
fig_log.write('\n')

[GPU_d, reg_choose, cell_count] = ReadMain('Suitability/main_reg.1', fig_log )

os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_d)

df_num = CountDF('Suitability',fig_log)

[cov , cov_without_nodata , df , df_without_nodata , type_count, size, cov_label, df_label] = Sample_change(sample1, sample2, df_num)



if reg_choose == 0:
    RandomForest(sample1, df_without_nodata, cov_without_nodata, df,  type_count, size, fig_log, cell_count)

fig_log.write('Finish at: ')
fig_log.write(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
fig_log.write('\n')