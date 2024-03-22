# 1.    使用tensorflow2.0完成以下操作（每小题10分）
# (1)    矩阵处理
import tensorflow as tf
# ①    使用标准正态分布，随机创建维度为2,8的矩阵
s=tf.random.normal([2,8])
print(s)
# ②    将维度变换为4，4
s=tf.reshape(s,[4,4])
print(s)
# ③    查看每一行的矩阵
for i in range(4):
    print(s[i])
# ④    查看每一列的最大值索引
print(tf.argmax(s,axis=0))
# ⑤    将矩阵第二行和第三行数据转置位置

zh=tf.gather(s,[0,2,1,3])
# (2)    求f(x) = a*x**2 + b*x + c的最小值
c1=tf.constant(1,dtype=tf.float32)
c2=tf.constant(1,dtype=tf.float32)
c3=tf.constant(1,dtype=tf.float32)
x=tf.Variable(10,dtype=tf.float32)
# ①    合理创建变量和常量
# ②    设定优化模型
# ③    循环迭代计算
# ④    打印相关数据值

op=tf.keras.optimizers.SGD(learning_rate=0.01)
for i in range(1000):
    with tf.GradientTape() as tape:
        y=c1*x**2+c2*x+c3
    td=tape.gradient(y,x)
    op.apply_gradients([(td,x)])
    tf.print(i,y,td,x)

