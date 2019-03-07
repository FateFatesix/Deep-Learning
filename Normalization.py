# 归一化：１）把数据变成(０，１)或者（-1,1）之间的
# 小数。主要是为了数据处理方便提出来的，把数据映射到0～
# 1范围之内处理，更加便捷快速。２）把有量纲表达式变成无
# 量纲表达式，便于不同单位或量级的指标能够进行比较和加权
# 。归一化是一种简化计算的方式，即将有量纲的表达式，经过变换，化为无量纲的表达式，成为纯量。
#归一化主要分为以下几种 Min-Max Normalization  平均归一化 非线性归一化 零均值归一化

def MaxMinNormalization(x,Max,Min):  
    x = (x - Min) / (Max - Min);  
    return x;  

def Zero_ScoreNormalization(x,mu,sigma):   # 零均值-归一化方差
    x = (x - mu) / sigma;  
    return x;  

# Keras中常见的一操作
# scaler = Normalizer().fit(X)        # Normalizer（）归一化处理后fit（）
# trainX = scaler.transform(X)        # transform：是将数据进行转换，比如数据的归一化，将测试数据按照训练数据同样的模型进行转换，得到特征向量。
# X_train = np.array(trainX)

# y_train1 = np.array(Y)
# y_train= to_categorical(y_train1)   #  to_categorical就是将类别向量转换为二进制（只有0和1）的矩阵类型表示
