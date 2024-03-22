# -*- coding: utf-8 -*-
# @Time    : 2024/3/22 10:59
# @Author  : 张壹
# @File    :day_2.py
# @Software: PyCharm
# 1.    使用tensorflow2.0完成以下操作（每小题10分）
import tensorflow as tf
# (1)    矩阵处理
# ①    使用标准正态分布，随机创建维度为2,8的矩阵
random_normal = tf.random.normal([2, 8])
# ②    将维度变换为4，4
tf_reshape = tf.reshape(random_normal, [4, 4])
# ③    查看每一行的矩阵
for i in range(4):
    print(tf_reshape[i])
# ④    查看每一列的最大值索引
print(tf.argmax(tf_reshape,axis=0))
# ⑤    将矩阵第二行和第三行数据转置位置
tf_gather = tf.gather(tf_reshape, [0, 2, 1, 3])
print(tf_gather)
# (2)    求f(x) = a*x**2 + b*x + c的最小值
# ①    合理创建变量和常量
a=tf.constant(2,dtype=tf.float32)
b=tf.constant(1,dtype=tf.float32)
c=tf.constant(1,dtype=tf.float32)
x=tf.Variable(10,dtype=tf.float32)

# ②    设定优化模型
op=tf.keras.optimizers.SGD(0.01)
# ③    循环迭代计算
for i in range(100):
    with tf.GradientTape() as tape:
        f_x = a * x ** 2 + b * x + c
    tape_gradient = tape.gradient(f_x, x)
    op.apply_gradients([(tape_gradient,x)])
# ④    打印相关数据值
    tf.print(i,f_x,tape_gradient,x)