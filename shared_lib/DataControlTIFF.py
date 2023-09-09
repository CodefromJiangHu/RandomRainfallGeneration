from osgeo import ogr, osr, gdal
import cv2
import numpy as np
import sys
import os
#import geopandas
# import numba
# from numba import jit,njit

# 2.求解索引处一定缓冲区内的最大值
def get_max_with_buffer(matrix_value, list_idx, buffer_size):
    """
    :param matrix_value: 特征矩阵，必须为矩阵
    :param list_idx: 样本点的行列索引
    :param buffer_size: bu方法的大小
    :return:
    """
    list_delta = np.arange(-1*buffer_size, buffer_size+1)
    list_results = matrix_value[list_idx]
    for i in range(len(list_delta)):
        new_rows_idx = list_idx[0]+list_delta[i]
        for j in range(len(list_delta)):
            new_cols_idx = list_idx[1]+list_delta[j]
            final_idx = new_rows_idx, new_cols_idx
            # 提取值
            list_results = np.amax([list_results, matrix_value[final_idx]], axis=0)
    return list_results
# 2.求解索引处一定缓冲区内的最小值
def get_min_with_buffer(matrix_value, list_idx, buffer_size):
    """
    :param matrix_value: 特征矩阵，必须为矩阵
    :param list_idx: 样本点的行列索引
    :param buffer_size: bu方法的大小
    :return:
    """
    list_delta = np.arange(-1*buffer_size, buffer_size+1)
    list_results = matrix_value[list_idx]
    for i in range(len(list_delta)):
        new_rows_idx = list_idx[0]+list_delta[i]
        for j in range(len(list_delta)):
            new_cols_idx = list_idx[1]+list_delta[j]
            final_idx = new_rows_idx, new_cols_idx
            # 提取值
            list_results = np.amin([list_results, matrix_value[final_idx]], axis=0)
    return list_results

# 2.求解索引处一定缓冲区内的平均值
def get_mean_with_buffer(matrix_value, list_idx, buffer_size):
    """
    :param matrix_value: 特征矩阵，必须为矩阵
    :param list_idx: 样本点的行列索引
    :param buffer_size: bu方法的大小
    :return:
    """
    list_delta = np.arange(-1*buffer_size, buffer_size+1)
    list_results = np.zeros_like(matrix_value[list_idx])
    for i in range(len(list_delta)):
        new_rows_idx = list_idx[0]+list_delta[i]
        for j in range(len(list_delta)):
            new_cols_idx = list_idx[1]+list_delta[j]
            final_idx = new_rows_idx, new_cols_idx
            # 提取值
            list_results = list_results + matrix_value[final_idx]
    return list_results/(len(list_delta)**2)

# 2.求解索引处一定缓冲区内的大于阈值的比例
def get_rate_upperThreshold_with_buffer(ipt_threshold, matrix_value, list_idx, buffer_size):
    """
    :param matrix_value: 特征矩阵，必须为矩阵
    :param list_idx: 样本点的行列索引
    :param buffer_size: bu方法的大小
    :return:
    """
    max_rows, max_cols = matrix_value.shape
    threshold = ipt_threshold
    num_upper = 0
    num_included = 0
    list_delta = np.arange(-1*buffer_size, buffer_size+1)
    list_results = np.zeros_like(matrix_value[list_idx])
    for i in range(len(list_delta)):
        new_rows_idx = list_idx[0]+list_delta[i]
        for j in range(len(list_delta)):
            new_cols_idx = list_idx[1]+list_delta[j]
            final_idx = new_rows_idx, new_cols_idx
            flag_low = np.logical_and((new_rows_idx >= 0),(new_cols_idx>=0))
            flag_up = np.logical_and((new_rows_idx <max_rows),(new_cols_idx<max_cols))
            if np.logical_and(flag_low,flag_up):
                num_included = num_included+1
            else:
                continue
            # 提取值
            tem_value = matrix_value[final_idx]
            if tem_value > threshold:
                num_upper = num_upper + 1
    return num_upper/num_included

# 平面坐标转换为行列号
def xy2RowCoum(input_MoniterList, cellsize, x_min, y_max):
    tem_Row_Coum_List = np.zeros([len(input_MoniterList[:,1]),3])
    for i in range(len(input_MoniterList[:,1])):
        # 编号相等
        tem_Row_Coum_List[i, 0] = input_MoniterList[i, 0]
        # 行号转化
        tem_X = input_MoniterList[i, 1]
        tem_Y = input_MoniterList[i, 2]
        tem_Row = int((y_max-tem_Y)/cellsize)
        tem_Coum = int((tem_X-x_min)/cellsize)
        # 数组赋值
        tem_Row_Coum_List[i, 1] = tem_Row
        tem_Row_Coum_List[i, 2] = tem_Coum
    return tem_Row_Coum_List
# 平面坐标转换为行列号索引
def xy2RowCoum_single(x,y, cellsize, x_min, y_max):
    # 行号转化
    tem_X = x
    tem_Y = y
    tem_Row = int((y_max-tem_Y)/cellsize)
    tem_Coum = int((tem_X-x_min)/cellsize)
    return tem_Row, tem_Coum

# 返回最大数组
def return_maxArray_(Array_a, Array_b):
    max = np.maximum(Array_a, Array_b)
    return max
# 返回最小数组
def return_minArray_(Array_a,Array_b):
    min = np.minimum(Array_a,Array_b)
    return min
# 返回最小数组
def return_SumArray_(Array_a,Array_b):
    a = np.array(Array_a)
    b = np.array(Array_b)
    return a+b

# 读取参考参考栅格图层，返回其行列宽度、投影、和同样大小的二维矩阵
def readTif(fileName):
    #dataset = gdal.Open('原始数据//dem//FillDEM1.0.tif')
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()
    #dataset = gdal.Open('dem/FillDEM1.0.tif')
    dataset = gdal.Open(fileName)
    im_width = dataset.RasterXSize #栅格矩阵的列数
    im_height = dataset.RasterYSize #栅格矩阵的行数
    im_bands = dataset.RasterCount #波段数
    im_geotrans = dataset.GetGeoTransform()
    band = dataset.GetRasterBand(1)

    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    return im_data, dataset

# 参考源码 https://blog.csdn.net/weixin_39190382/article/details/106441762
def load_img_to_array(img_file_path):
    """
    读取栅格数据，将其转换成对应数组
    :param: img_file_path: 栅格数据路径
    :return: 返回投影，几何信息，和转换后的数组(5888,5888,10)
    """
    dataset = gdal.Open(img_file_path)  # 读取栅格数据
    # print('处理图像的栅格波段数总共有：', dataset.RasterCount)
    # 判断是否读取到数据
    if dataset is None:
        print('Unable to open *.tif')
        sys.exit(1)  # 退出
    projection = dataset.GetProjection()  # 投影
    transform = dataset.GetGeoTransform()  # 几何信息
    # 直接读取dataset
    img_array = dataset.ReadAsArray()
    return projection, transform, img_array

def get_pro_from_img(img_file_path):
    """
    读取栅格数据，将其转换成对应数组
    :param: img_file_path: 栅格数据路径
    :return: 返回投影，几何信息，和转换后的数组(5888,5888,10)
    """
    dataset = gdal.Open(img_file_path)  # 读取栅格数据
    # print('处理图像的栅格波段数总共有：', dataset.RasterCount)
    # 判断是否读取到数据
    if dataset is None:
        print('Unable to open *.tif')
        sys.exit(1)  # 退出
    projection = dataset.GetProjection()  # 投影
    transform = dataset.GetGeoTransform()  # 几何信息
    # 直接读取dataset
    # img_array = dataset.ReadAsArray()
    return projection, transform
def get_array_from_img(img_file_path):
    print("正在从本地电脑载入文件："+str(img_file_path))
    """
    读取栅格数据，将其转换成对应数组
    :param: img_file_path: 栅格数据路径
    :return: 返回投影，几何信息，和转换后的数组(5888,5888,10)
    """
    dataset = gdal.Open(img_file_path)  # 读取栅格数据
    # print('处理图像的栅格波段数总共有：', dataset.RasterCount)
    # 判断是否读取到数据
    if dataset is None:
        print('Unable to open *.tif')
        sys.exit(1)  # 退出
    # projection = dataset.GetProjection()  # 投影
    # transform = dataset.GetGeoTransform()  # 几何信息
    # 直接读取dataset
    img_array = dataset.ReadAsArray()
    return img_array

# 根据对应关系 更新二维数组的单元值
def update_values_according_to_rela(list_oldValue, list_newValue, matrix_oldValue):
    """
    :param list_oldValue: 矩阵本来的值 序列
    :param list_newValue: 矩阵新的值 序列
    :param matrix_oldValue: 矩阵
    :return:
    """
    m, n = np.array(matrix_oldValue).shape
    final_matrix = np.zeros(shape=[m, n], dtype="float")
    result = np.zeros(shape=[m, n], dtype="float")
    length = len(list_oldValue)
    for i in range(len(list_oldValue)):
        flag_matrix = (matrix_oldValue == list_oldValue[i])
        final_matrix = flag_matrix * list_newValue[i]
        result = result + final_matrix
        print("转换完成度（%）："+str(np.round(100*i/length)))
    return result

# 根据对应关系 更新二维数组的单元值
# ID值与被赋值的值域之间不存在交集时使用
def update_values_according_to_rela_NchangeNodata(list_oldValue, list_newValue, matrix_oldValue):
    """
    :param list_oldValue: 矩阵本来的值 序列
    :param list_newValue: 矩阵新的值 序列
    :param matrix_oldValue: 矩阵
    :return:
    """
    m, n = np.array(matrix_oldValue).shape
    final_matrix = np.zeros(shape=[m, n], dtype="float")
    result = np.zeros(shape=[m, n], dtype="float")
    length = len(list_oldValue)
    for i in range(len(list_oldValue)):
        matrix_oldValue = np.where(matrix_oldValue == list_oldValue[i], list_newValue[i], matrix_oldValue)
        print("转换完成度（%）："+str(np.round(100*i/length)))
    result = matrix_oldValue
    return result


# 参考源码 https://blog.csdn.net/weixin_39190382/article/details/106441762
# 矩阵转化为TIF
def predit_to_tif(mat, projection, tran, mapfile):
    """
    将数组转成tif文件写入硬盘
    :param mat: 数组
    :param projection: 投影信息
    :param tran: 几何信息
    :param mapfile: 文件路径
    :return:
    """

    row = mat.shape[0]  # 矩阵的行数
    columns = mat.shape[1]  # 矩阵的列数
    geo_transform = tran

    # print(geo_transform)

    # dim_z = mat.shape[2]  # 通道数
    dim_z = 1

    driver = gdal.GetDriverByName('GTiff')  # 创建驱动
    # 创建文件
    # dst_ds = driver.Create(mapfile, columns, row, dim_z, gdal.GDT_UInt16)
    dst_ds = driver.Create(mapfile, columns, row, dim_z, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(geo_transform)  # 设置几何信息
    dst_ds.SetProjection(projection)  # 设置投影

    # 将数组的各通道写入tif图片
    map = mat[:, :]
    dst_ds.GetRasterBand(1).WriteArray(map)
    # for channel in np.arange(dim_z):
    #     map = mat[:, :, channel]
    #     dst_ds.GetRasterBand(int(channel + 1)).WriteArray(map)

    dst_ds.FlushCache()  # 写入硬盘
    dst_ds = None

# 矩阵转化为TIF
def matrix_to_tif(mat, projection, tran, celltype, mapfile):
    """
    将数组转成tif文件写入硬盘
    :param mat: 数组
    :param projection: 投影信息
    :param tran: 几何信息
    :param mapfile: 文件路径
    :param celltype: 值类型  float or int
    :return:
    """
    row = mat.shape[0]  # 矩阵的行数
    columns = mat.shape[1]  # 矩阵的列数
    geo_transform = tran
    # print(geo_transform)
    # dim_z = mat.shape[2]  # 通道数
    dim_z = 1
    driver = gdal.GetDriverByName('GTiff')  # 创建驱动
    # 创建文件
    if celltype=="float":
        dst_ds = driver.Create(mapfile, columns, row, dim_z, gdal.GDT_Float32)
    elif celltype=="int":
        dst_ds = driver.Create(mapfile, columns, row, dim_z, gdal.GDT_UInt16)
    else:
        print("未设置值类型或设置字符串存在错误！")
        return
    dst_ds.SetGeoTransform(geo_transform)  # 设置几何信息
    dst_ds.SetProjection(projection)  # 设置投影
    # 将数组的各通道写入tif图片
    map = mat[:, :]
    band = dst_ds.GetRasterBand(1)
    band.SetNoDataValue(-9999)
    dst_ds.GetRasterBand(1).WriteArray(map)

    # for channel in np.arange(dim_z):
    #     map = mat[:, :, channel]
    #     dst_ds.GetRasterBand(int(channel + 1)).WriteArray(map)

    dst_ds.FlushCache()  # 写入硬盘
    dst_ds = None

# 矩阵转化为TIF
def gis_asc_to_tif(projection, tran, celltype, ascfile, mapfile):
    """
    将数组转成tif文件写入硬盘
    :param mat: 数组
    :param projection: 投影信息
    :param tran: 几何信息
    :param mapfile: 文件路径
    :param celltype: 值类型  float or int
    :return:
    """
    # asc 文件解析
    # writeMessage("ncols         " + str(ncols), filePath)
    # writeMessage("nrows         " + str(nrows), filePath)
    # writeMessage("xllcorner     " + str(xllcorner), filePath)
    # writeMessage("yllcorner     " + str(yllcorner), filePath)
    # writeMessage("cellsize      " + str(cellsize), filePath)
    # writeMessage("NODATA_value  " + str(NODATA_value), filePath)
    # 获取左上角x 和 y坐标
    num = 6 # 读取前6行
    str_head = []
    with open(ascfile) as rows_ascfile:
        for i in range(num):
            tem_row = str(rows_ascfile.readline()).strip() # 读取两端剔除空格的字符串
            list_str = tem_row.split(" ")
            item_tem = list_str[-1] # 取list的最后一个元素，即相应的头文件值
            str_head.append(item_tem)
            # print(tem_row)
    # 根据读取结果取值
    ncols = int(str_head[0])
    nrows = int(str_head[1])
    xllcorner = float(str_head[2])
    yllcorner = float(str_head[3])
    cellsize = float(str_head[4])
    NODATA_value = float(str_head[5])
    xltcorner = xllcorner
    yltcorner = yllcorner + cellsize * nrows
    # 转换位置信息更改为asc文件制定的脚点
    new_tran = list(tran)
    new_tran[0] = xltcorner
    new_tran[3] = yltcorner
    geo_transform = new_tran

    # 矩阵提取
    ascii_grid = np.loadtxt(ascfile, skiprows=6)
    row = ascii_grid.shape[0]  # 矩阵的行数
    columns = ascii_grid.shape[1]  # 矩阵的列数

    # print(geo_transform)
    # dim_z = mat.shape[2]  # 通道数
    dim_z = 1
    driver = gdal.GetDriverByName('GTiff')  # 创建驱动
    # 创建文件
    if celltype=="float":
        dst_ds = driver.Create(mapfile, columns, row, dim_z, gdal.GDT_Float32)
    elif celltype=="int":
        dst_ds = driver.Create(mapfile, columns, row, dim_z, gdal.GDT_UInt16)
    else:
        print("未设置值类型或设置字符串存在错误！")
        return
    dst_ds.SetGeoTransform(geo_transform)  # 设置几何信息
    # 坐上角坐标更新
    dst_ds.SetProjection(projection)  # 设置投影

    # 将数组的各通道写入tif图片
    map = ascii_grid[:, :]
    band = dst_ds.GetRasterBand(1)
    # band.SetNoDataValue(-9999)
    band.SetNoDataValue(NODATA_value )
    dst_ds.GetRasterBand(1).WriteArray(map)

    # for channel in np.arange(dim_z):
    #     map = mat[:, :, channel]
    #     dst_ds.GetRasterBand(int(channel + 1)).WriteArray(map)

    dst_ds.FlushCache()  # 写入硬盘
    dst_ds = None

# 读取所有小流域数据
def ReadVectorFile(fliepath):

    strVectorFile =fliepath
    #ds = ogr.Open("Watershed_DaduRiver_Mended.shp",False)
    # 注册所有的驱动
    ogr.RegisterAll()

    #打开数据
    ds = ogr.Open(strVectorFile, 0)

    # 获取该数据源中的图层个数，一般shp数据图层只有一个，如果是mdb、dxf等图层就会有多个
    iLayerCount = ds.GetLayerCount()

    # 获取第一个图层
    oLayer = ds.GetLayerByIndex(0)
    if oLayer == None:
           print("获取第%d个图层失败！\n", 0)
           return

    # 对图层进行初始化，如果对图层进行了过滤操作，执行这句后，之前的过滤全部清空
    oLayer.ResetReading()


    # 通过指定的几何对象对图层中的要素进行筛选
    #oLayer.SetSpatialFilter()

    # 通过指定的四至范围对图层中的要素进行筛选
    #oLayer.SetSpatialFilterRect()

    # 获取图层中的属性表表头并输出
    print("属性表结构信息：")
    oDefn = oLayer.GetLayerDefn()
    iFieldCount = oDefn.GetFieldCount()
    for iAttr in range(iFieldCount):
           oField = oDefn.GetFieldDefn(iAttr)
           print( "%s: %s(%d.%d)" % ( \
                             oField.GetNameRef(),\
                             oField.GetFieldTypeName(oField.GetType() ), \
                             oField.GetWidth(),\
                             oField.GetPrecision()))
           #oField.GetGeometryRef()
    # 输出图层中的要素个数
    print("要素个数 = %d", oLayer.GetFeatureCount(0))


    oFeature = oLayer.GetNextFeature()
    #下面开始遍历图层中的要素
    while oFeature is not None:
        print("当前处理第%d个: \n属性值：", oFeature.GetFID())
        # 获取要素中的属性表内容
        for iField in range(iFieldCount):
               oFieldDefn = oDefn.GetFieldDefn(iField)
               line =  " %s (%s) = " % ( \
                                  oFieldDefn.GetNameRef(),\
                                  ogr.GetFieldTypeName(oFieldDefn.GetType()))

               if oFeature.IsFieldSet( iField ):
                        line = line+ "%s" % (oFeature.GetFieldAsString( iField ) )
               else:
                        line = line+ "(null)"

               print(line)

        # 获取要素中的几何体
        oGeometry =oFeature.GetGeometryRef()

        # 为了演示，只输出一个要素信息
        break

        print(oGeometry)

def writeMessage(message_txt,txt_filePathName):
    filepath = txt_filePathName # ‘xxx.txt’
    if os.path.exists(filepath) == False:
        # 创建文件
        file = open(filepath, 'w', encoding='utf-8')
        file.close()
    file = open(filepath, 'r+', encoding='utf-8')
    # 读取一下数据，调整指针到末尾
    file.read()
    file.write(message_txt + "\n")
    file.close()
# 更新文件的后缀名为asc
def update_txt_with_asc(filePathName):
    if filePathName.endswith('.txt'):
        new_filePathName = filePathName[:-4] + ".asc"
        os.replace(filePathName, new_filePathName)
# 实现将numpy的二维矩阵转换GIS系统下的ascii文件
def convert_into_ascii_from_matrix(filePath,matrix_value,ncols,nrows,xllcorner,yllcorner,cellsize,NODATA_value):
    # 矩阵生成响应的.asc
    # 头文件写入
    writeMessage("ncols         " + str(ncols), filePath)
    writeMessage("nrows         " + str(nrows), filePath)
    writeMessage("xllcorner     " + str(xllcorner), filePath)
    writeMessage("yllcorner     " + str(yllcorner), filePath)
    writeMessage("cellsize      " + str(cellsize), filePath)
    writeMessage("NODATA_value  " + str(NODATA_value), filePath)
    # 矩阵写入
    for i in range(len(matrix_value)):
        list = matrix_value[i, :].tolist()
        str_row = str(list).strip("[]") # 去除数组左右两边的 括号
        list_str = str_row.split(",")
        str_row = " ".join(list_str)
        # 逗号改为空格

        writeMessage(str_row, filePath)
    # 改写文件后缀
    update_txt_with_asc(filePath)
    print(filePath)