import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import shuffle
import scipy
print("TF Version:", tf.__version__)
path = "/home/ghauch/Documents/ML_Submetering/Final_Data.csv"
path_y_hot = "/home/ghauch/Documents/ML_Submetering/Y_One_Hot.csv"
path_test_data = "/home/ghauch/Documents/ML_Submetering/Test_Data.csv"
path_y_hot_test = "/home/ghauch/Documents/ML_Submetering/Y_One_Hot_Test.csv"
path_test_time_series = "/home/ghauch/Documents/ML_Submetering/Test_Time_Series.csv"

# Loading Training Data
df = pd.read_csv(path,sep=';',header=None)
data = df.values
df_y_hot = pd.read_csv(path_y_hot,sep=';',header=None)
y_one_hot = df_y_hot.values

# Loading Test Data
df_test_data = pd.read_csv(path_test_data,sep=';',header=None)
test_data    = df_test_data.values
Num_Test_Customers = np.shape(test_data)[1]
shape_test = [np.shape(test_data)[0],Num_Test_Customers]                    # Test data shapes
df_y_hot_test = pd.read_csv(path_y_hot_test,sep=';',header=None)
y_one_hot_test = df_y_hot_test.values
test_data_X    = test_data[1:np.shape(test_data)[0],:] # Extract time series
test_data_Y    = test_data[0,:]                        # Extract Labels


# Test data full time series for final evaluation of sugggested method
df_test_time_series = pd.read_csv(path_test_time_series,sep=';',header=None)
test_time_series = df_test_time_series.values
test_data_Y_time_series = test_time_series[0,:]


y_one_hot_test_time_series = np.zeros([5,np.shape(test_time_series)[1]])
for k in range(np.shape(test_time_series)[1]):
    for p in range(5):
        if p+1 == test_time_series[0,k]:
            y_one_hot_test_time_series[p,k] = 1
shape_test_time_series = [np.shape(test_time_series)[0],np.shape(test_time_series)[1]]   


y_data = data[0,:]
X_data  = np.asarray(data[1:np.shape(data)[0],:])

batchsize = 16
learning_rate = 0.001
dropout = 0.95
N1 = 100
N2 = 100

num_samples = np.shape(X_data)[1]
input_size = np.shape(X_data)[0]
shape_test = [np.shape(test_data_X)[0],np.shape(test_data_X)[1]]
num_classes = 5
N = input_size
Nbatches = batchsize
epsilon = 1e-8


# Transposing datasets for correct input shape to the network
X_train =  np.transpose(X_data[:,:])
y_train = np.transpose(y_one_hot[:,:])
X_val   = np.transpose(test_data_X[:,:])
y_val   = np.transpose(y_one_hot_test[:,:])
y_data_confusion = test_data_Y
# Transformation of the time series 

def RecurrencePlot(inputArray,N):

    R = np.zeros([np.shape(inputArray)[1],N,N,1])
    radius = 1
    for k in range(np.shape(inputArray)[1]):
        for i in range(N):
            for j in range(N):
                if np.absolute(inputArray[i,k]-inputArray[j,k] < radius):
                    R[k,i,j,0] = np.absolute(inputArray[i,k]-inputArray[j,k])
    return R
# Vanilla Recurrence Plot
print("Calculating Recurrence plots")
R_train = RecurrencePlot(X_data,N)
# Loop for calculating the Recurrence Plots
#radius = 1
#for k in range(num_samples):
 #   for i in range(N):
  #      for j in range(N):
   #         if np.absolute(X_data[i,k]-X_data[j,k]) < radius:
    #            R_train[k,i,j,0] = np.absolute(X_data[i,k]-X_data[j,k])  

R_val = RecurrencePlot(test_data_X,N)
# Loop for calculating the Recurrence Plots
#radius = 1
#for k in range(shape_test[1]):
 #   for i in range(N):
  #      for j in range(N):
   #         if np.absolute(test_data_X[i,k]-test_data_X[j,k]) < radius:
    #            R_val[k,i,j,0] = np.absolute(test_data_X[i,k]-test_data_X[j,k])  


print("Done Calculating the Recurrance Plots")


# Placeholder for datavectors
X = tf.placeholder(tf.float32, [None,N,N,1]) 
Y = tf.placeholder(tf.float32, [None,num_classes])
initializer = tf.contrib.layers.xavier_initializer()

# Writers

# Define scalar summary


# Define Convolutional Layers


def conv2d(input_tensor,weights,bias,s=1):
    out = tf.nn.conv2d(input_tensor,weights,strides=[1,s,s,1],padding='SAME')
    out = tf.nn.bias_add(out,bias)
    return tf.nn.relu(out)

def maxpool2d(X,k,s):
    out = tf.nn.max_pool(X,ksize=[1,k,k,1],strides=[1,s,s,1],padding='SAME')
    return out

#Defining Network

def ConvNet(X,weights,biases):
    with tf.name_scope("Conv_Layer1"):
        # Layer 1
        conv1 = conv2d(X,weights['filter1'],biases['bias_filter1'])
        conv1 = maxpool2d(conv1,2,2)
        shape1 = np.shape(conv1)

        batch_mean_conv_1,batch_variance_conv_1 = tf.nn.moments(conv1,[0])
        scale_conv_1 = tf.Variable(tf.ones([shape1[1],shape1[2],shape1[3]]))
        beta_conv_1  = tf.Variable(tf.ones([shape1[1],shape1[2],shape1[3]]))

        conv1 = tf.nn.batch_normalization(conv1,batch_mean_conv_1,batch_variance_conv_1,beta_conv_1,scale_conv_1,epsilon) 
    with tf.name_scope("Conv_Layer2"):    
        # Layer 2
        conv2 = conv2d(conv1,weights['filter2'],biases['bias_filter2'])
        conv2 = maxpool2d(conv2,2,2)

        shape2 = np.shape(conv2)    

        batch_mean_conv_2,batch_variance_conv_2 = tf.nn.moments(conv2,[0])
        scale_conv_2 = tf.Variable(tf.ones([shape2[1],shape2[2],shape2[3]]))
        beta_conv_2  = tf.Variable(tf.ones([shape2[1],shape2[2],shape2[3]]))

        conv2 = tf.nn.batch_normalization(conv2,batch_mean_conv_2,batch_variance_conv_2,beta_conv_2,scale_conv_2,epsilon) 
    with tf.name_scope("MLP_Layer1"):
        # MLP - Layer
        
        mlp_input = tf.reshape(conv2,[-1, np.shape(conv2)[1]*np.shape(conv2)[2]*np.shape(conv2)[3]])

        #Layer one

        fc1 = tf.matmul(mlp_input,weights['wh1'])
        #Batch normalization
        batch_mean1,batch_variance1 = tf.nn.moments(fc1,[0])
        scale1 = tf.Variable(tf.ones([N1]))
        beta1 = tf.Variable(tf.zeros([N1]))

        fc1 = tf.nn.batch_normalization(fc1,batch_mean1,batch_variance1,beta1,scale1,epsilon) 
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1,dropout)
    with tf.name_scope("MLP_Layer2"):
        # Fully connected layer 2
        fc2 = tf.matmul(fc1,weights['wh2'])
        batch_mean2,batch_variance2 = tf.nn.moments(fc2,[0])
        scale2 = tf.Variable(tf.ones([N2]))
        beta2 = tf.Variable(tf.zeros([N2]))

        fc2 = tf.nn.batch_normalization(fc2,batch_mean2,batch_variance2,beta2,scale2,epsilon) 
        fc2 = tf.nn.relu(fc2)
        fc2 = tf.nn.dropout(fc2,dropout)
    # Return outputs 
    with tf.name_scope("Classification_Layer"):
        pred = tf.nn.bias_add(tf.matmul(fc2,weights['out']),biases['bias_out'])
    return pred


# Define the weights 

# Starting with simple 2-layer CNN

weights = {
    'filter1': tf.get_variable('filter1',[3,3,1,32],initializer = tf.contrib.layers.xavier_initializer_conv2d()),
    'filter2': tf.get_variable('filter2',[3,3,32,64],initializer = tf.contrib.layers.xavier_initializer_conv2d()),
    #Fully connected Layer
    'wh1':      tf.get_variable('wh1',[2304,N1],initializer = tf.contrib.layers.xavier_initializer()),
    'wh2':      tf.get_variable('wh2',[N1,N2],initializer = tf.contrib.layers.xavier_initializer()),
    'out':      tf.get_variable('out',[N2,num_classes],initializer = tf.contrib.layers.xavier_initializer())
}

biases = {
'bias_filter1': tf.Variable(initializer([32])),
'bias_filter2': tf.Variable(initializer([64])),
'bias_out':     tf.Variable(initializer([num_classes]))  
}




# Training network


forward_pass = ConvNet(X,weights,biases)
prediction   = tf.nn.softmax(forward_pass)
pred_number = tf.argmax(prediction,1)+1
# Loss and optimizer
with tf.name_scope("Loss"):
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = forward_pass, labels = Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1 = 0.9 , beta2= 0.999, epsilon=1e-8)
    train_op = optimizer.minimize(loss_op)
    tf.summary.scalar("Loss",loss_op)
# Model Evaluation

with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    tf.summary.scalar("Accuracy",accuracy)
# Training

init = tf.global_variables_initializer()
tf.set_random_seed(2015)
val_loss = []
val_acc = []
train_loss = []
train_acc = []

merged = tf.summary.merge_all()
writer_scalar = tf.summary.FileWriter("/home/ghauch/Documents/ML_Submetering/Logs")
writer = tf.summary.FileWriter("/home/ghauch/Documents/ML_Submetering/Logs")


with tf.Session() as sess:
    sess.run(init)
    for epoch in range(2):
        print("Epoch "+str(epoch))

        for i in range(Nbatches):
            batch_X = R_train[i*batchsize:(i+1)*batchsize,:,:,:]
            batch_y =y_train[i*batchsize:(i+1)*batchsize,:]
            sess.run(train_op,feed_dict={X: batch_X, Y: batch_y})
        valloss,valacc = sess.run([loss_op,accuracy],feed_dict={X: R_val, Y: y_val})
        trainloss,trainacc = sess.run([loss_op,accuracy],feed_dict={X: R_train, Y: y_train})
        validation_log = sess.run(merged,feed_dict={X: R_val, Y: y_val})
        writer_scalar.add_summary(validation_log,epoch)
        print(" Valdiation Loss = " +"{:.4f}".format(valloss) + ", Validation Accuracy = " + "{:.3f}".format(valacc))
        print(" Training Loss = " +"{:.4f}".format(trainloss) + ", Training Accuracy = " + "{:.3f}".format(trainacc))
        
        val_loss.append(valloss)
        val_acc.append(valacc)
        train_loss.append(trainloss)
        train_acc.append(trainacc)

    print("Optimization Finished")
    predictions = sess.run(pred_number,feed_dict={X: R_val})
    confusion_matrix = sess.run(tf.confusion_matrix(y_data_confusion,predictions))
    print("Confusion Matrix")
    print(confusion_matrix)
    # TensorBoard Filewriter 

    # FINAL TEST ACCURACY CHECKING 
    print("Rank of Test matrix")
    print(np.linalg.matrix_rank(test_time_series))
    print("Starting To Compute The final test accuracy")
    pred = []
    correct_test_prediction = []
    booleans    = test_time_series[:,:] >=  0
    booleans2   = test_time_series[:,:] ==  0
    time_points = input_size
    print(test_data_Y_time_series)
    print(shape_test_time_series[1])



    for j in range(shape_test_time_series[1]):   # For every time series
        prediction_number = np.zeros(5)
        count =  0   
                              # Counts Number of Chunks, make prediction only if count is larger than one so as to ignore arrays with only zeros
        for i in range(int((shape_test_time_series[0])/time_points)): # label is included hence -1
            
            # maxnumber of slices
            if np.all(booleans[i*time_points:i*time_points+time_points,j]) == True and np.sum(booleans2[i*time_points:i*time_points+time_points,j]) < time_points/2:

                
                feed_data = test_time_series[i*time_points:i*time_points+time_points,j]

                if np.var(feed_data) > 0: # take only non-constant time-series
                    maxval = np.max(feed_data,axis=-1)
                    
                    #normalization
                    feed_data = feed_data / maxval 
                    feed_data = RecurrencePlot([feed_data - np.mean(feed_data)],N)
                    print(np.shape(feed_data))
                    temp_num = sess.run(pred_number,feed_dict={X: [feed_data]})
                                
                    prediction_number[temp_num-1] = prediction_number[temp_num-1] + 1 
                    count = count+1
        if count > 1:
            #print(prediction_number)
            print(prediction_number)
            final_prediction = np.argmax(prediction_number)+1
            #print(final_prediction)
            #print(final_prediction)
            if final_prediction == test_data_Y_time_series[j]:
                correct_test_prediction.append(1)
            else:
                correct_test_prediction.append(0)  
        
    plt.plot(correct_test_prediction)
    plt.show()
    final_test_accuracy = np.mean(correct_test_prediction)
    print("Final Accuracy is "+ str(final_test_accuracy*100)+"%")
writer.add_graph(sess.graph)  