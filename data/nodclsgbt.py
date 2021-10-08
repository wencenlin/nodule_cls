import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mahotas
from mahotas.features.lbp import lbp

# -----此程式為用來獲取基於 GBM、結節直徑和結節像素的性能-----
# 預處理後，我們將 CT 切片轉換為 3D numpy 數組，並獲得帶有 posfix '_label.npy' 的新結節註釋。
# 結節註釋是結節的中心和結節的直徑。 您可以使用它從預處理的 3D numpy 數組中裁剪結節以進行結節分類任務。
CROPSIZE = 32  # 24#30#36
print(CROPSIZE)
pdframe = pd.read_csv('annotationdetclsconvfnl_v3.csv',
                      names=['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant'])
srslst = pdframe['seriesuid'].tolist()[1:]
crdxlst = pdframe['coordX'].tolist()[1:]
crdylst = pdframe['coordY'].tolist()[1:]
crdzlst = pdframe['coordZ'].tolist()[1:]
dimlst = pdframe['diameter_mm'].tolist()[1:]
mlglst = pdframe['malignant'].tolist()[1:]

newlst = []
import csv

fid = open('annotationdetclsconvfnl_v3.csv', 'r')
# writer = csv.writer(fid)
# writer.writerow(['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm', 'malignant'])
for i in range(len(srslst)):
    # writer.writerow([srslst[i] + '-' + str(i), crdxlst[i], crdylst[i], crdzlst[i], dimlst[i], mlglst[i]])
    # newlst.append([srslst[i] + '-' + str(i), crdxlst[i], crdylst[i], crdzlst[i], dimlst[i], mlglst[i]])
    newlst.append([srslst[i], crdxlst[i], crdylst[i], crdzlst[i], dimlst[i], mlglst[i]])
fid.close()

preprocesspath = 'D:/luna16/preprocess/all/'
savepath = 'D:/luna16/crop_v3/'
import os
import os.path

if not os.path.exists(savepath): os.mkdir(savepath)
# for idx in range(len(newlst)):
for idx in range(1):
    fname = newlst[idx][0]  # 1.3.6.1.4.1.14519.5.2.1.6279.6001.262736997975960398949912434623-0-0-0-0-0-0-0-0-0-0-0-0
    # if fname != '1.3.6.1.4.1.14519.5.2.1.6279.6001.119209873306155771318545953948-581': continue
    pid = fname.split('-')[0]  # 1.3.6.1.4.1.14519.5.2.1.6279.6001.262736997975960398949912434623
    crdx = int(float(newlst[idx][1]))  # 140
    crdy = int(float(newlst[idx][2]))  # 101
    crdz = int(float(newlst[idx][3]))  # 43
    dim = int(float(newlst[idx][4]))  # 4
    data = np.load(os.path.join(preprocesspath, pid + '_clean.npy'))  # 經mask處理完的肺部影像
    bgx = int(max(0, crdx - CROPSIZE / 2))  # 140-32/2=140-16=124
    bgy = int(max(0, crdy - CROPSIZE / 2))  # 101-32/2=101-16=85
    bgz = int(max(0, crdz - CROPSIZE / 2))  # 43-32/2=43-16=27
    cropdata = np.ones((CROPSIZE, CROPSIZE, CROPSIZE)) * 170  # 先設32*32*32的背景(水=170)
    cropdatatmp = np.array(data[0, bgx:bgx + CROPSIZE, bgy:bgy + CROPSIZE, bgz:bgz + CROPSIZE])  # data[0, 124:156, 85:117, 27:59]
    cropdata[CROPSIZE // 2 - cropdatatmp.shape[0] // 2:CROPSIZE // 2 - cropdatatmp.shape[0] // 2 + cropdatatmp.shape[0],
    CROPSIZE // 2 - cropdatatmp.shape[1] // 2:CROPSIZE // 2 - cropdatatmp.shape[1] // 2 + cropdatatmp.shape[1],
    CROPSIZE // 2 - cropdatatmp.shape[2] // 2:CROPSIZE // 2 - cropdatatmp.shape[2] // 2 + cropdatatmp.shape[2]] = np.array(2 - cropdatatmp)  # 我還沒理解2 - cropdatatmp
    assert cropdata.shape[0] == CROPSIZE and cropdata.shape[1] == CROPSIZE and cropdata.shape[2] == CROPSIZE
    np.save(os.path.join(savepath, fname + '.npy'), cropdata)


# -----------------------------以上為製造crop的data----------------------------
# train use gbt
# subset1path = 'D:/luna16/data_subset/subset1/'
# testfnamelst = []
# for fname in os.listdir(subset1path):
#     if fname.endswith('.mhd'):
#         testfnamelst.append(fname[:-4])
# ntest = 0
# for idx in range(len(newlst)):
#     fname = newlst[idx][0]
#     if fname.split('-')[0] in testfnamelst: ntest += 1
# print('ntest', ntest, 'ntrain', len(newlst) - ntest)
#
# traindata = np.zeros((len(newlst) - ntest, CROPSIZE * CROPSIZE * CROPSIZE))
# trainlabel = np.zeros((len(newlst) - ntest,))
# testdata = np.zeros((ntest, CROPSIZE * CROPSIZE * CROPSIZE))
# testlabel = np.zeros((ntest,))
#
# trainidx = testidx = 0
# for idx in range(len(newlst)):
#     fname = newlst[idx][0]
#     # print fname
#     data = np.load(os.path.join(savepath, fname + '.npy'))
#     # print data.shape
#     bgx = int(data.shape[0] / 2 - CROPSIZE / 2)
#     bgy = int(data.shape[1] / 2 - CROPSIZE / 2)
#     bgz = int(data.shape[2] / 2 - CROPSIZE / 2)
#     data = np.array(data[bgx:bgx + CROPSIZE, bgy:bgy + CROPSIZE, bgz:bgz + CROPSIZE])
#     if fname.split('-')[0] in testfnamelst:
#         testdata[testidx, :] = np.reshape(data, (-1,)) / 255
#         # testdata[testidx, -4] = newlst[idx][1]
#         # testdata[testidx, -3] = newlst[idx][2]
#         # testdata[testidx, -2] = newlst[idx][3]
#         # testdata[testidx, -1] = newlst[idx][4]
#         testlabel[testidx] = newlst[idx][-1]
#         testidx += 1
#     else:
#         traindata[trainidx, :] = np.reshape(data, (-1,)) / 255
#         # traindata[trainidx, -4] = newlst[idx][1]
#         # traindata[trainidx, -3] = newlst[idx][2]
#         # traindata[trainidx, -2] = newlst[idx][3]
#         # traindata[trainidx, -1] = newlst[idx][4]
#         trainlabel[trainidx] = newlst[idx][-1]
#         trainidx += 1
# maxtraindata1 = max(traindata[:, -1])
# # traindata[:, -1] = np.array(traindata[:, -1] / maxtraindata1)
# # maxtraindata2 = max(traindata[:, -2])
# # traindata[:, -2] = np.array(traindata[:, -2] / maxtraindata2)
# # maxtraindata3 = max(traindata[:, -3])
# # traindata[:, -3] = np.array(traindata[:, -3] / maxtraindata3)
# # maxtraindata4 = max(traindata[:, -4])
# # traindata[:, -4] = np.array(traindata[:, -4] / maxtraindata4)
# # testdata[:, -1] = np.array(testdata[:, -1] / maxtraindata1)
# # testdata[:, -2] = np.array(testdata[:, -2] / maxtraindata2)
# # testdata[:, -3] = np.array(testdata[:, -3] / maxtraindata3)
# # testdata[:, -4] = np.array(testdata[:, -4] / maxtraindata4)
# from sklearn.ensemble import GradientBoostingClassifier as gbt
#
#
# def gbtfunc(dep):
#     m = gbt(max_depth=dep, random_state=0)
#     m.fit(traindata, trainlabel)
#     predtrain = m.predict(traindata)
#     predtest = m.predict_proba(testdata)
#     # print predtest.shape, predtest[1,:]
#     return np.sum(predtrain == trainlabel) / float(traindata.shape[0]), \
#            np.mean((predtest[:, 1] > 0.5).astype(int) == testlabel), predtest  # / float(testdata.shape[0]),
#
#
# # trainacc, testacc, predtest = gbtfunc(3)
# # print trainacc, testacc
# # np.save('pixradiustest.npy', predtest[:,1])
# from multiprocessing import Pool
#
# p = Pool(2)
# acclst = p.map(gbtfunc, range(1, 9, 1))  # 3,4,1))#5,1))#1,9,1))
# for acc in acclst:
#     print("{0:.4f}".format(acc[0]), "{0:.4f}".format(acc[1]))
# p.close()
# # for dep in xrange(1,9,1):
# # 	m = gbt(max_depth=dep)
# # 	m.fit(traindata, trainlabel)
# # 	print dep, 'trainacc', np.sum(m.predict(traindata) == trainlabel) / float(traindata.shape[0])
# # 	print dep, 'testacc', np.sum(m.predict(testdata) == testlabel) / float(testdata.shape[0])
