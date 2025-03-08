# 给kvasir数据图像进行排序，需要排序的文件有testdataset中的images和masks，result中的kvasir
import os

path_list = "./300/TestDataset/CVC-300-TV/images"
class_list = ".png"

file_in = os.listdir(path_list)
num_file_in = len(file_in)

for i in range(0, num_file_in):
    t = str(i + 700)
    new_name = os.rename(path_list + "/" + file_in[i], path_list + "/" + t +class_list)

file_out = os.listdir(path_list)
print(file_out)
