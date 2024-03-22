# 1.使用tensorflow2.0完成以下操作（每小题10分）
import tensorflow as tf
# (1)矩阵处理
# ①使用标准正态分布，随机创建维度为2,8的矩阵
tf_normal = tf.random.normal([2, 8])
# ②将维度变换为4，4
tf_reshape = tf.reshape(tf_normal, [4, 4])
# ③查看每一行的矩阵
for i in range(4):
    print(tf_reshape[i])
# ④查看每一列的最大值索引
print(tf.argmax(tf_reshape, axis=0))
# ⑤将矩阵第二行和第三行数据转置位置
g = tf.gather(tf_reshape, [0, 2, 1, 3])
# (2)求f(x) = a*x**2 + b*x + c的最小值
# ①合理创建变量和常量
a = tf.constant(2, dtype=tf.float32)
b = tf.constant(1, dtype=tf.float32)
c = tf.constant(1, dtype=tf.float32)
x = tf.Variable(10.0, dtype=tf.float32)
# ②设定优化模型
op = tf.keras.optimizers.SGD(learning_rate=0.01)
# ③循环迭代计算
# ④打印相关数据值
for i in range(1000):
    with tf.GradientTape() as tape:
        h = a * x ** 2 + b * x + c
    delet = tape.gradient(h, x)
    op.apply_gradients([(delet, x)])
    tf.print(i, h, delet, x)
