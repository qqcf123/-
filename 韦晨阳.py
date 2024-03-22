# 1.使用tensorflow2.0完成以下操作（每小题10分）
import tensorflow as tf
# (1)矩阵处理
# ①使用标准正态分布，随机创建维度为2,8的矩阵
tr=tf.random.normal([2,8])
# ②将维度变换为4，4
tr_=tf.reshape(tr,[4,4])
# ③查看每一行的矩阵
for i in range(4):
    print(tr_[i])
# ④查看每一列的最大值索引
print(tf.argmax(tr_,axis=0))
# ⑤将矩阵第二行和第三行数据转置位置
g=tf.gather(tr_,[0,2,1,3])
# (2)求f(x) = a*x**2 + b*x + c的最小值
# ①合理创建变量和常量
a=tf.constant(1,dtype=tf.float32)
b=tf.constant(1,dtype=tf.float32)
c=tf.constant(1,dtype=tf.float32)
x=tf.Variable(10,dtype=tf.float32)
# ②设定优化模型
op=tf.keras.optimizers.SGD(0.01)
# ③循环迭代计算
for i in range(1000):
    with tf.GradientTape() as tape:
        f_x=a*x**2+b*x+c
        g_x=tape.gradient(f_x,x)#梯度
        op.apply_gradients([(g_x,x)])#梯度下降
        tf.print(i,f_x,x)

# ④打印相关数据值