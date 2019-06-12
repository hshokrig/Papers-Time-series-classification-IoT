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

print( np.any(np.isnan(data)))
## IN THIS CODE I WILL DO A SIMPLE TWO HIDDEN LAYER NEURAL NETWORK 2HLNN
#print("Test set")
#print(test_data_Y)
# Learning parameter333s
learning_rate = 0.001
batchsize = 8
dropout = 0.80
N1 = 10
N2 = 50
N3 = 50
N4 = 50
N5 = 50
N6 = 10

num_samples = np.shape(X_data)[1]
input_size  = np.shape(X_data)[0]
num_classes = 5
Nbatches    = int(num_samples/batchsize)
epsilon     = 1e-8



# Transposing datasets for correct input shape to the network
X_train =  np.transpose(X_data[:,:])
y_train = np.transpose(y_one_hot[:,:])
X_val   = np.transpose(test_data_X[:,:])
y_val   = np.transpose(y_one_hot_test[:,:])
y_data_confusion = test_data_Y

print("Amount of training data is: "+str(np.shape(X_train)[0]))
# Placeholder for datavectors
X = tf.placeholder(tf.float32, [None,input_size]) 
Y = tf.placeholder(tf.float32, [None,num_classes])
initializer = tf.contrib.layers.xavier_initializer()
# Define the network

def mlp(X, weights, biases,dropout):
    with tf.name_scope("Layer_1"):
        #Layer One
        fc1 = tf.matmul(X,weights['wh1'])
        batch_mean1,batch_variance1 = tf.nn.moments(fc1,[0])
        fc1 = tf.nn.batch_normalization(fc1,batch_mean1,batch_variance1,biases['beta1'],biases['scale1'],epsilon) 
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1,dropout)
    with tf.name_scope("Layer_2"):
        # Layer Two
        fc2 = tf.matmul(fc1,weights['wh2'])
        batch_mean2,batch_variance2 = tf.nn.moments(fc2,[0])
        fc2 = tf.nn.batch_normalization(fc2,batch_mean2,batch_variance2,biases['beta2'],biases['scale2'],epsilon) 
        fc2 = tf.nn.relu(fc2)
        fc2 = tf.nn.dropout(fc2,dropout)
    with tf.name_scope("Layer_3"):
        # Layer Three
        fc3 = tf.matmul(fc2,weights['wh3'])
        batch_mean3,batch_variance3 = tf.nn.moments(fc3,[0])
        fc3 = tf.nn.batch_normalization(fc3,batch_mean3,batch_variance3,biases['beta3'],biases['scale3'],epsilon) 
        fc3 = tf.nn.relu(fc3)
        fc3 = tf.nn.dropout(fc3,dropout)
    with tf.name_scope("Layer_4"):
        # Layer Four
        fc4 = tf.matmul(fc3,weights['wh4'])
        batch_mean4,batch_variance4 = tf.nn.moments(fc4,[0])
        fc4 = tf.nn.batch_normalization(fc4,batch_mean4,batch_variance4,biases['beta4'],biases['scale4'],epsilon) 
        fc4 = tf.nn.relu(fc4)
        fc4 = tf.nn.dropout(fc4,dropout)
    with tf.name_scope("Layer_5"):
        # Layer Five
        fc5 = tf.matmul(fc4,weights['wh5'])
        batch_mean5,batch_variance5 = tf.nn.moments(fc5,[0])
        fc5 = tf.nn.batch_normalization(fc5,batch_mean5,batch_variance5,biases['beta5'],biases['scale5'],epsilon) 
        fc5 = tf.nn.relu(fc5)
        fc5 = tf.nn.dropout(fc5,dropout)
    with tf.name_scope("Layer_6"):
        # Layer Six
        fc6 = tf.matmul(fc5,weights['wh6'])
        batch_mean6,batch_variance6 = tf.nn.moments(fc6,[0])
        fc6 = tf.nn.batch_normalization(fc6,batch_mean6,batch_variance6,biases['beta6'],biases['scale6'],epsilon) 
        fc6 = tf.nn.relu(fc6)
        fc6 = tf.nn.dropout(fc6,dropout)


# Return outputs
    with tf.name_scope("Output_Layer"):
        pred = tf.nn.bias_add(tf.matmul(fc6,weights['out']),biases['biout'])

    return pred

# Set weights and biases
weights = {
    # First  hidden layer
    'wh1': tf.Variable(initializer([input_size,N1])),
    # Second hidden layer
    'wh2': tf.Variable(initializer([N1,N2])),
    # Third hidden layer
    'wh3': tf.Variable(initializer([N2,N3])),
    # Fourth Hidden Layer
    'wh4': tf.Variable(initializer([N3,N4])),
    # Fifth Hidden Layer
    'wh5': tf.Variable(initializer([N4,N5])),
    # Sixth Hidden Layer
    'wh6': tf.Variable(initializer([N5,N6])),
    # Output layer
    'out': tf.Variable(initializer([N6,num_classes]))
}

biases = {
    'scale1': tf.Variable(initializer([N1])),
    'beta1' : tf.Variable(initializer([N1])),

    'scale2': tf.Variable(initializer([N2])),
    'beta2' : tf.Variable(initializer([N2])),

    'scale3': tf.Variable(initializer([N3])),
    'beta3' : tf.Variable(initializer([N3])),

    'scale4': tf.Variable(initializer([N4])),
    'beta4' : tf.Variable(initializer([N4])),

    'scale5': tf.Variable(initializer([N5])),
    'beta5' : tf.Variable(initializer([N5])),

    'scale6': tf.Variable(initializer([N6])),
    'beta6' : tf.Variable(initializer([N6])),

    'biout': tf.Variable(initializer([num_classes]))
}


# Model & Evaluation
with tf.name_scope("Logits"):
    mlp_model = mlp(X,weights,biases,dropout) # Feeds data through model defined above
    prediction = tf.nn.softmax(mlp_model) # Constructs a prediction
    pred_number = tf.argmax(prediction,1)+1


# Loss and Optimizer
with tf.name_scope("Loss"):
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = mlp_model, labels = Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1 = 0.9 , beta2= 0.999, epsilon=1e-8)
    train_op = optimizer.minimize(loss_op)
    tf.summary.scalar("Validation_Loss",loss_op)
# Model Evaluation
with tf.name_scope("Model_Eval"):
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
writer_scalar = tf.summary.FileWriter("/home/ghauch/Documents/ML_Submetering/Logs_NN")
writer = tf.summary.FileWriter("/home/ghauch/Documents/ML_Submetering/Logs_NN")

# SESSION

with tf.Session() as sess:
    sess.run(init)
    print("Optimization Started")
    for epoch in range(15):
        print("Epoch "+str(epoch))

        for i in range(Nbatches):
            batch_X = X_train[i*batchsize:(i+1)*batchsize,:]
            batch_y =y_train[i*batchsize:(i+1)*batchsize,:]
            sess.run(train_op,feed_dict={X: batch_X, Y: batch_y})
        valloss,valacc = sess.run([loss_op,accuracy],feed_dict={X: X_val, Y: y_val})
        trainloss,trainacc = sess.run([loss_op,accuracy],feed_dict={X: X_train, Y: y_train})
        
        print(" Valdiation Loss = " +"{:.4f}".format(valloss) + ", Validation Accuracy = " + "{:.3f}".format(valacc))
        print(" Training Loss = " +"{:.4f}".format(trainloss) + ", Training Accuracy = " + "{:.3f}".format(trainacc))

        val_loss.append(valloss)
        val_acc.append(valacc)
        train_loss.append(trainloss)
        train_acc.append(trainacc)
        # Tensorboard logging
        validation_log = sess.run(merged,feed_dict={X: X_val, Y: y_val})
        writer_scalar.add_summary(validation_log,epoch)


    print("Optimization Finished")
   
    predictions = sess.run(pred_number,feed_dict={X: X_val})
    #print(predictions)
    confusion_matrix = sess.run(tf.confusion_matrix(y_data_confusion,predictions))
    print("Confusion Matrix")
    print(confusion_matrix)


    # FINAL TEST ACCURACY CHECKING 
    print("Rank of Test matrix")
    print(np.linalg.matrix_rank(test_time_series))
    print("Starting To Compute The final test accuracy")
    pred = []
    correct_test_prediction = []
    booleans    = test_time_series[:,:] >=  0
    booleans2   = test_time_series[:,:] ==  0
    time_points = input_size
   # print(test_data_Y_time_series)
    #print(shape_test_time_series[1])



    for j in range(shape_test_time_series[1]):   # For every time series
        prediction_number = np.zeros(5)
        count =  0   
                        # Counts Number of Chunks, make prediction only if count is larger than one so as to ignore arrays with only zeros
        for i in range(int((shape_test_time_series[0])/time_points)): # label is included hence -1
            if count < 50:
                # maxnumber of slices
                if np.all(booleans[i*time_points:i*time_points+time_points,j]) == True and np.sum(booleans2[i*time_points:i*time_points+time_points,j]) < time_points/2:

                    
                    feed_data = test_time_series[i*time_points:i*time_points+time_points,j]
                
                    if np.var(feed_data) > 0: # take only non-constant time-series
                        maxval = np.max(feed_data,axis=-1)
                        
                        #normalization
                        feed_data = feed_data / maxval 
                        feed_data = feed_data - np.mean(feed_data)
            
                        temp_num = sess.run(pred_number,feed_dict={X: [feed_data]})
                                    
                        prediction_number[temp_num-1] = prediction_number[temp_num-1] + 1 
                        count = count+1
        if count > 1:
            #pridnt(prediction_number)
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
plt.figure(1)
plt.plot(val_loss,'r')
plt.plot(train_loss,'g')
plt.show()








