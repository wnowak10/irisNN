import tensorflow as tf
import pandas as pd
from sklearn.cross_validation import train_test_split


iris = pd.read_csv("/Users/wnowak/Desktop/iris.csv")

# print(iris.columns.values)
# drop unneccessary ID column
iris = iris.drop('Id', 1)

# this is super convenient. pops the target column from train set.  and creates target
target = iris.pop("Species")

ir=iris.values


train_labels = pd.get_dummies(target)
trrr=train_labels.values

X_train,X_test,y_train,y_test = train_test_split(ir,trrr,test_size=0.2)
# # so no we have predictors and y values, separated into test and train
y_train



# print(X_train.shape)

# there are four descriptor variables
x = tf.placeholder(tf.float32, [None, 4])
# # take 4 descriptors to one of 3 outputs
W = tf.Variable(tf.zeros([4, 3]))
# # add biases for each output
b = tf.Variable(tf.zeros([3]))

# #implement model. these are predicteds
y = tf.nn.softmax(tf.matmul(x, W) + b)

# #hold correct results
y_ = tf.placeholder(tf.float32, [None, 3])

# CE loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# GD training
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# this is where preicted y's go
tf_softmax_correct = tf.placeholder("float", [None,3])


tf_correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(tf_softmax_correct,1))
tf_accuracy = tf.reduce_mean(tf.cast(tf_correct_prediction, "float"))

# # init all vars
init = tf.initialize_all_variables()



sess = tf.Session()
sess.run(init)


# this works
k=[]
for i in range(1000):
  sess.run(train_step, feed_dict={x: X_train, y_: y_train})
# Print accuracy
  result = sess.run(tf_accuracy, feed_dict={x: X_test, tf_softmax_correct: y_test})
  print("Run {},{}".format(i,result))
  k.append(result)


result = sess.run(tf_accuracy, feed_dict={x: X_test, 
                                          tf_softmax_correct: y_test})
print("Result: {}".format(result))

ans = sess.run(y, feed_dict={x: X_test})
print(y_test[0:3])
print("Correct prediction\n",ans[0:3])

