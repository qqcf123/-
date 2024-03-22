# # 1.使用tensorflow2.0完成以下操作（每小题10分）
# # (1)矩阵创建
# # ①创建一个维度为3*3的全1矩阵
# import tensorflow as tf
# tf_one = tf.ones([3,3])
# # ②使用range，创建一个1-9的1阶张量
# tf_range = tf.range(start=1,limit=10)
# # ③打印上题的维度
# print(tf_range.shape)
# # ④将上题维度修改为3,1,3
# tf_reshape = tf.reshape(tf_range,[3,1,3])
# # ⑤使用函数，去除维度中函数1的维度
# tf_squ = tf.squeeze(tf_range)
# # (2)切片及其他
# # ①使用1-9的向量，使用切片，打印3,4,5,6  1，2，3，4，5，6，7，8，9 a[2:5]
# tf_sp = tf_range[2:6]
# # ②打印上题向量的均值
# print(tf.reduce_mean(tf_sp))
# # ③创见一个2行2列的标准正态分布矩阵
# tf_random = tf.random.normal([2,2])
# # ④创建一个2行2列的全0矩阵
# tf_zero = tf.zeros([2,2])
# # ⑤将3,4问的结果拼接成一个4行2列的结果
# tf_con = tf.concat([tf_random,tf_zero],axis=0)










# 1.使用tensorflow2.0完成以下操作（每小题10分）
# (1)矩阵处理
# ①使用标准正态分布，随机创建维度为2,8的矩阵
import tensorflow as tf
tf_random = tf.random.normal([2,8])
# ②将维度变换为4，4
tf_reshape = tf.reshape(tf_random,[4,4])
# ③查看每一行的矩阵
for i in range(4):
    print(tf_reshape[i])
# ④查看每一列的最大值索引
print(tf.argmax(tf_reshape,axis=0))
# ⑤将矩阵第二行和第三行数据转置位置

# (2)求f(x) = a*x**2 + b*x + c的最小值
# ①合理创建变量和常量
a = tf.constant(2,dtype=tf.float32)
b = tf.constant(1,dtype=tf.float32)
c = tf.constant(1,dtype=tf.float32)
x = tf.Variable(10,dtype=tf.float32)
# ②设定优化模型
op = tf.keras.optimizers.SGD(learning_rate=0.01)
# ③循环迭代计算
for i in range(1000):
    with tf.GradientTape() as tape:
        f_x = a * x ** 2 + b * x + c
    df_x = tape.gradient(f_x,x)
    op.apply_gradients([(df_x,x)])
# ④打印相关数据值
    tf.print(i,f_x,x)













