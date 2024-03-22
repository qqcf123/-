# 1.使用tensorflow2.0完成以下操作（每小题10分）
import tensorflow as tf

# (1)矩阵处理
# ①使用标准正态分布，随机创建维度为2,8的矩阵
normal = tf.random.normal(shape=[2, 8])

# ②将维度变换为4，4
tf_reshape = tf.reshape(normal, shape=[4, 4])

# ③查看每一行的矩阵
for i in range(len(tf_reshape)):
    tf.print(tf_reshape[i, :])

# ④查看每一列的最大值索引
tf.print(tf.argmax(tf_reshape, axis=0))

# ⑤将矩阵第二行和第三行数据转置位置
gather_matrix = tf.gather(tf_reshape, [0, 2, 1, 3])

# (2)求f(x) = a*x**2 + b*x + c的最小值
# ①合理创建变量和常量
x = tf.Variable(initial_value=10, dtype=tf.float32)
a = tf.constant(2, dtype=tf.float32)
b = tf.constant(1, dtype=tf.float32)
c = tf.constant(1, dtype=tf.float32)

# ②设定优化模型
# tf.keras.optimizers用于定义和应用优化算法的模块。其中，SGD代表随机梯度下降（Stochastic Gradient Descent）算法。
op = tf.keras.optimizers.SGD(learning_rate=0.01)

# ③循环迭代计算
for i in range(1000):
    # tf.GradientTape()是TensorFlow中可用于自动计算梯度的上下文管理器。
    with tf.GradientTape() as tape:
        f_x = a * x ** 2 + b * x + c

    df_x = tape.gradient(f_x, x)

    # op.apply_gradients()是用于将计算得到的梯度应用于模型参数的方法。它接受一个梯度和变量的元组列表，并根据梯度更新变量的值。
    op.apply_gradients([(df_x, x)])
    # ④打印相关数据值
    tf.print(i, 'f(x):', f_x, 'df(x):', df_x, 'x:', x)

