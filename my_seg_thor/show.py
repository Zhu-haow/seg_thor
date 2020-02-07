import matplotlib.pyplot as plt
import numpy as np

'''
    查看数据集原图和标签
'''

path1 = r'..\data\data_npy\Patient_25\Patient_25_100_image.npy'
path2 = r'..\data\data_npy\Patient_25\Patient_25_100_label.npy'
img = np.load(path1)
label = np.load(path2)
plt.figure()
plt.imshow(img, cmap = 'gray')
plt.figure()
plt.imshow(label, cmap = 'gray')
plt.show()