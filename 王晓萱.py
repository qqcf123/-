# 1.使用tensorflow2.0完成以下操作（每小题10分）
import tensorflow as tf
import random
# (1)矩阵创建
# ①创建一个维度为3*3的全1矩阵
ones = tf.ones(shape=[3, 3])
print(ones)
# ②使用range，创建一个1-9的1阶张量
tf_range = tf.range(start=1, limit=10)
print(tf_range)
# ③打印上题的维度
print(tf_range.shape)
# ④将上题维度修改为3,1,3
reshape = tf.reshape(tf_range, [3, 1, 3])
print(reshape)
# ⑤使用函数，去除维度中函数1的维度
squeeze = tf.squeeze(tf_range)
print(squeeze)
# (2)切片及其他
# ①使用1-9的向量，使用切片，打印3,4,5,6  1，2，3，4，5，6，7，8，9 a[2:5]
range1 = tf_range[2:6]
print(range1)
# ②打印上题向量的均值
print(tf.reduce_mean(range1))
# ③创见一个2行2列的标准正态分布矩阵
normal = tf.random.normal(shape=[2, 2])
print(normal)
# ④创建一个2行2列的全0矩阵
zeros = tf.zeros(shape=[2, 2])
print(zeros)
# ⑤将3,4问的结果拼接成一个4行2列的结果
concat = tf.concat([normal, zeros], axis=0)
print(concat)
print(concat.shape)





#
# # 1.使用tensorflow2.0完成以下操作（每小题10分）
# import tensorflow as tf
# # (1)矩阵创建
# # ①创建一个维度为3*3的全1矩阵
# zeros = tf.zeros(shape=[3, 3])
# print(zeros)
# # ②使用range，创建一个1-9的1阶张量
# tf_range = tf.range(start=1, limit=10)
# print(tf_range)
# # ③打印上题的维度
# print(tf_range.shape)
# # ④将上题维度修改为3,1,3
# reshape = tf.reshape(tf_range,shape=[3, 1, 3])
# print(reshape)
# # ⑤使用函数，去除维度中函数1的维度
# squeeze = tf.squeeze(reshape)
# print(squeeze)
# # (2)切片及其他
# ①使用1-9的向量，使用切片，打印3,4,5,6  1，2，3，4，5，6，7，8，9 a[2:5]
# range_ = tf_range[2:6]
# print(range_)
# # ②打印上题向量的均值
# print(tf.reduce_mean(range_))
# # ③创见一个2行2列的标准正态分布矩阵
# normal = tf.random.normal(shape=[2, 2])
# print(normal)
# # ④创建一个2行2列的全0矩阵
# tf_zeros = tf.zeros(shape=[2, 2])
# print(tf_zeros)
# # ⑤将3,4问的结果拼接成一个4行2列的结果
# concat = tf.concat([normal, tf_zeros],axis=0)
# print(concat)


# 1.使用tensorflow2.0完成以下操作（每小题10分）
import tensorflow as tf
# (1)矩阵创建
# ①创建一个维度为3*3的全1矩阵
ones = tf.ones(shape=[3, 3])
print(ones)
# ②使用range，创建一个1-9的1阶张量
o=tf.range(start=1,limit=10)
print(o)
# ③打印上题的维度
print(o.shape)
# ④将上题维度修改为3,1,3
reshape = tf.reshape(o,[3, 1, 3])
print(reshape)
# ⑤使用函数，去除维度中函数1的维度
squeeze = tf.squeeze(reshape)
print(squeeze)
# # (2)切片及其他
# # ①使用1-9的向量，使用切片，打印3,4,5,6  1，2，3，4，5，6，7，8，9 a[2:5]
#
# # ②打印上题向量的均值
# print(tf.reduce_mean(range_))
# ③创见一个2行2列的标准正态分布矩阵
normal = tf.random.normal([2, 2])
print(normal)
# ④创建一个2行2列的全0矩阵
zeros = tf.zeros([2, 2])
print(zeros)
# ⑤将3,4问的结果拼接成一个4行2列的结果
concat = tf.concat([normal, zeros], axis=0)
print(concat)