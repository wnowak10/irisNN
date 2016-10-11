# #iris2
# rely on https://www.oreilly.com/learning/hello-tensorflow
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

# to classify iris species
# adapted from https://www.oreilly.com/learning/hello-tensorflow

iris = pd.read_csv("/Users/wnowak/Desktop/iris.csv")

# print(iris.columns.values)
# drop unneccessary ID column
iris = iris.drop('Id', 1)

# this is super convenient. pops the target column from train set.  and creates target
target = iris.pop("Species")

# converts from dataframe to np array
ir=iris.values

# convert train labels to one hots
train_labels = pd.get_dummies(target)
# make np array
trrr=train_labels.values

# x_train,x_test,y_train,y_test = train_test_split(ir,trrr,test_size=0.33,dtype=float)
x_train,x_test,y_train,y_test = train_test_split(ir,trrr,test_size=0.33)
# # so now we have predictors and y values, separated into test and train

#set np array d type

x_train,x_test,y_train,y_test = np.array(x_train,dtype='float32'), np.array(x_test,dtype='float32'),np.array(y_train,dtype='float32'),np.array(y_test,dtype='float32')


# placeholders
x = tf.placeholder("float", [None, 4])
w = tf.Variable(tf.zeros([4, 3]))
# # add biases for each output
b = tf.Variable(tf.zeros([3]))
y_= tf.placeholder("float", [None,3])

y=tf.nn.softmax(tf.matmul(x, w) + b)
loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(.235).minimize(loss)


# check accuracy
tf_correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
tf_accuracy = tf.reduce_mean(tf.cast(tf_correct_prediction, "float"))

# for value in [x, w, b, y, y_, loss]:
#     tf.scalar_summary(value.op.name, value)

# summaries = tf.merge_all_summaries()

sess = tf.Session()
# summary_writer = tf.train.SummaryWriter('iris', sess.graph)

sess.run(tf.initialize_all_variables())
for i in range(500):
    # summary_writer.add_summary(sess.run(summaries), i)
    sess.run(train_step,feed_dict={x: x_train, y_: y_train})


result = sess.run(tf_accuracy, feed_dict={x: x_test, 
                                          y_: y_test})
print("Result: {}".format(result))

ans = sess.run(y, feed_dict={x: x_test})
print(y_test[0:3])
print("Correct prediction\n",ans[0:3])
