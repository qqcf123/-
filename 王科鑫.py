# 1.	使用tensorflow2.0完成以下操作（每小题10分）
# (1)	矩阵创建
import tensorflow as tf
# ①	创建一个维度为3*3的全1矩阵
tf_pred = tf.ones([3,3])
# ②	使用range，创建一个1-9的1阶张量
tf_range = tf.range(1,10)
# ③	打印上题的维度
print(tf_range.shape)
# ④	将上题维度修改为3,1,3
tf_reshape = tf.reshape(tf_range,[3,1,3])
# ⑤	使用函数，去除维度中函数1的维度
tf_sq = tf.squeeze(tf_reshape)
print(tf_sq)
# (2)	切片及其他
# ①	使用1-9的向量，使用切片，打印3,4,5,6  1，2，3，4，5，6，7，8，9 a[2:5]
aa = tf_range[2:6]
print(aa)
# ②	打印上题向量的均值
mean = tf.reduce_mean(aa)
print(mean)
# ③	创见一个2行2列的标准正态分布矩阵
bb = tf.random.normal([2,2])
print(bb)
# ④	创建一个2行2列的全0矩阵
zreo = tf.zeros([2,2])
# ⑤	将3,4问的结果拼接成一个4行2列的结果
cc = tf.concat([bb,zreo],0)
print(cc)
print(cc.shape)
