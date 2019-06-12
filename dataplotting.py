import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import scipy
import os

def using_indexed_assignment(x):
    "https://stackoverflow.com/a/5284703/190597 (Sven Marnach)"
    result = np.empty(len(x), dtype=int)
    temp = x.argsort()
    result[temp] = np.arange(len(x))
    return result


path = "/home/ghauch/Documents/ML_Submetering/Concatenated_File_total.csv"
#Hard Threshholding for removing outliers
df = pd.read_csv(path,sep=';',header=None)
data = df.values
shape = np.shape(data)
print(shape)
# Shuffeling the data

index  = np.arange(0,shape[1])
choice = np.random.choice(index,shape[1],replace=False)
print(choice)
data   = data[:,choice]
print(data)
print("Number of Time Series is:" + str(np.shape(data)[1]))
print( np.any(np.isnan(data)))
max_length = np.shape(data)[0]


training_percentage = 0.7
threshhold = 2000                                                       # Hard Threshholding?
time_points = 24                                                            # 720 is one month 9000 a year
indices = data[:,:] < threshhold
data[indices == False] = -1
# Splitting into test data
data_test = data[:,int(training_percentage*shape[1]):shape[1]] 


# Training Data
y = data[0,0:int(training_percentage*shape[1])]
X  = data[1:shape[0],0:int(training_percentage*shape[1])]

#Test Data                                                                           # Shall be written directly to file or not, not
X_test = data[1:max_length,int(training_percentage*shape[1]):shape[1]]
y_test = data[0,int(training_percentage*shape[1]):shape[1]]

# Next we create the chunks of size time_points which will be stored as columns in X_resampled with corresponding label in y_resampled FOR TRAINING SET

number_of_chunks_per_customer = []
X_resampled = []
y_resampled = []
left_out_slice = []
booleans    = X[:,:] >=  0
booleans2   = X[:,:] ==  0
boolvec = []
for j in range(np.shape(X)[1]):                                                    # Looping over the samples
    label = y[j]
    count = 0
    for i in range(int(np.shape(X)[0] / time_points)):  # Looping over the number of chunks assuming no overlap
        # In this statement we take the chunks where all values are greater or equal to zero(I.e excluding the NaN) and where there are at most 25% values equal to zero.
        #Then append them in X_resampled
        if count < 20:                        # Change this to change the number of snippets from each time-series
            if  np.all(booleans[i*time_points:i*time_points+time_points,j]) == True and np.sum(booleans2[i*time_points:i*time_points+time_points,j]) < time_points/2:
                if np.var(X[i*time_points:i*time_points+time_points,j]) > 0:
                    if count > 10:
                        var = np.var(X[i*time_points:i*time_points+time_points,j]) # Adding some augmentation to the data random noise
                    
                        X_resampled.append(X[i*time_points:i*time_points+time_points,j] + np.random.normal(0,np.sqrt(var),np.shape(X[i*time_points:i*time_points+time_points,j])))
                        y_resampled.append(label)
                    else:
                        # One pure sample from each device
                        X_resampled.append(X[i*time_points:i*time_points+time_points,j])
                        y_resampled.append(label)
                    count = count+1
            else:
                # Filling NaN's with interpolation if the number of NaN's per slice is below some limit
                if np.sum(booleans[i*time_points:i*time_points+time_points,j]) > 21 and np.sum(booleans2[i*time_points:i*time_points+time_points,j]) < time_points/2: # maximum of 4 Nans

                    nan_indices = np.where( booleans[i*time_points:i*time_points+time_points,j] == False )
                    for k in nan_indices:
                        X[i*time_points+k,j] = np.mean(X[i*time_points:i*time_points+time_points,j]) # setting Nan's to mean value

                    X_resampled.append(X[i*time_points:i*time_points+time_points,j])
                    y_resampled.append(label)
                    count = count+1
                #  Monitor the Slices that are left out
                if np.any(booleans[i*time_points:i*time_points+time_points,j]) == True and np.sum(booleans2[i*time_points:i*time_points+time_points,j]) < time_points/2 :   # If anyone is not nan
                    left_out_slice.append(X[i*time_points:i*time_points+time_points,j])
                    boolvec.append(time_points-np.sum(booleans[i*time_points:i*time_points+time_points,j]))

                
    number_of_chunks_per_customer.append(count)


plt.hist(boolvec)
plt.title("Histogram of number of NaN's per slice with the maximum nuber of zeros equal to "+str(int(time_points/2)))
plt.show()
left_out_slice = np.stack(left_out_slice,axis=-1)
plt.plot(number_of_chunks_per_customer)
plt.title("Distribution of Number of Chunks on Customers/Meters")
plt.show()


## Does the same as above but for test dataset
number_of_chunks_per_customer_test = []
X_resampled_test = []
y_resampled_test = []
booleans_test    = X_test[:,:] >=  0
booleans2_test   = X_test[:,:] ==  0
for j in range(np.shape(X_test)[1]):                                           # Looping over the samples
    label_test = y_test[j]
    count = 0
    for i in range(int(np.shape(X_test)[0] / time_points)):  # Looping over the number of chunks assuming no overlap
        # In this statement we take the chunks where all values are greater or equal to zero(I.e excluding the NaN) and where there are at most 25% values equal to zero.
        #Then append them in X_resampled
        if count < 1:    
            if  np.all(booleans_test[i*time_points:i*time_points+time_points,j]) == True and np.sum(booleans2_test[i*time_points:i*time_points+time_points,j]) < time_points/2: 
                if np.var(X_test[i*time_points:i*time_points+time_points,j]) > 0:  # Avoid having all the same inputs, want to capture some patterns
                    
                    X_resampled_test.append(X_test[i*time_points:i*time_points+time_points,j])
                    count = count+1
                    y_resampled_test.append(label_test)
    number_of_chunks_per_customer_test.append(count)


#Next we build the matrix
X_resampled = np.stack(X_resampled,axis=-1)  # Stacked data
X_resampled_test = np.stack(X_resampled_test,axis=-1)  # Stacked data
#Putting together in one big matrix so it is easier to shuffle.  
X_big = np.concatenate([[y_resampled],X_resampled],axis=0)
X_big_test = np.concatenate([[y_resampled_test],X_resampled_test],axis=0)
# Get indexes for different classes
print(np.shape(X_big))
print(np.shape(X_big_test))


X_big = X_big[:,np.argsort(X_big[0,:])]
X_big_test = X_big_test[:,np.argsort(X_big_test[0,:])]

print(X_big[0,:])


a,return_index,return_counts = np.unique(X_big[0,:], return_index=True, return_counts=True)


print("There are a total of " +str(np.shape(X_resampled)[1]) +" samples distributed as follows.")
print(return_counts)
print(return_index)

print()
print()
a_test,return_index_test,return_counts_test = np.unique(X_big_test[0,:] ,return_index=True, return_counts=True)
print("The number of samples in each class in test set is distributed as follows" )
print(return_counts_test)
print(return_index_test)
print()
#Building final dataset of min_number_of_samples per class, randomly sampled and shuffeled
min_number_of_samples = np.min(return_counts) # The number of samples from the smallest class
min_number_of_samples_test = np.min(return_counts_test)

indexvector = np.arange(0,np.shape(X_big)[1]) # Indices of all samples
indexvector_test = np.arange(0,np.shape(X_big_test)[1])
data_shuffeled = np.zeros([time_points+1,5*min_number_of_samples]) #Preallocate memory for final data array.
data_shuffeled_test = np.zeros([time_points+1,5*min_number_of_samples_test])


for i in range(5):
    #Picks min_number_of_samples random indices from each class and puts them in the final data array.
   indexchoice = np.random.choice(indexvector[return_index[i]:return_index[i]+return_counts[i]],min_number_of_samples,replace=False) 

   indexchoice_test = np.random.choice(indexvector_test[return_index_test[i]:return_index_test[i]+return_counts_test[i]],min_number_of_samples_test,replace=False)

   data_shuffeled[:,i*min_number_of_samples:i*min_number_of_samples+min_number_of_samples]                       = X_big[:,indexchoice]

   data_shuffeled_test[:,i*min_number_of_samples_test:i*min_number_of_samples_test+min_number_of_samples_test]   = X_big_test[:,indexchoice_test]

print( np.any(np.isnan(data_shuffeled)))
print(np.any(np.isnan(data_shuffeled)))
# For checking that this works
a2,return_index2,return_counts2 = np.unique(data_shuffeled[0,:], return_index=True, return_counts=True)  


a2_test,return_index2_test,return_counts2_test = np.unique(data_shuffeled_test[0,:],return_index = True, return_counts = True)

print("The number of samples in each class is " +str(min_number_of_samples)+".")
print(return_counts2)
print(return_index2)
print()
print()

print("The number of samples in each class for Test set is " +str(min_number_of_samples_test)+".")
print(return_counts2_test)
print(return_index2_test)
print()
print()


X_big_test = data_shuffeled_test               # For simplicity
#normalization_mean_array=[]
#normalization_std_array=[]
# Normalization
# Looping over the columns, normalizing each sample to values between -1 and 1.
for k in range(np.shape(data_shuffeled)[1]):
    normalization_constant_max = np.max(data_shuffeled[1:np.shape(data_shuffeled)[0],k],axis=-1)
   
    data_shuffeled[1:np.shape(data_shuffeled)[0],k] = data_shuffeled[1:np.shape(data_shuffeled)[0],k] / normalization_constant_max
    data_shuffeled[1:np.shape(data_shuffeled)[0],k] = data_shuffeled[1:np.shape(data_shuffeled)[0],k]-np.mean(data_shuffeled[1:np.shape(data_shuffeled)[0],k])
for i in range(np.shape(X_big_test)[1]):
    normalization_constant_max = np.max(X_big_test[1:np.shape(X_big_test)[0],i],axis=-1)
   
    X_big_test[1:np.shape(X_big_test)[0],i] = X_big_test[1:np.shape(X_big_test)[0],i] / normalization_constant_max
    X_big_test[1:np.shape(X_big_test)[0],i] = X_big_test[1:np.shape(X_big_test)[0],i] - np.mean(X_big_test[1:np.shape(X_big_test)[0],i])
      
#Makes sense to normalize the data?

#plt.figure(1)
#plt.plot(normalization_mean_array)
#plt.show()
#plt.figure(2)
#plt.plot(normalization_std_array)
#plt.show()

print( np.any(np.isnan(data_shuffeled)))
print(np.any(np.isnan(X_big_test)))
# Final shuffle of the training data
indexvector2 = np.arange(0,np.shape(data_shuffeled)[1])

indexchoice2 = np.random.choice(indexvector2, len(indexvector2),replace=False)
print(indexchoice2)

data_shuffeled = data_shuffeled[:,indexchoice2]


# Extract final data, write to csv?
y_final = data_shuffeled[0,:]
X_final  = data_shuffeled[1:np.shape(data_shuffeled)[0],:]

y_one_hot = np.zeros([5,np.shape(data_shuffeled)[1]])
for k in range(np.shape(data_shuffeled)[1]):
    for p in range(5):
        if p+1 == y_final[k]:
            y_one_hot[p,k] = 1
print(np.shape(y_one_hot)[1])

y_one_hot_test = np.zeros([5,np.shape(X_big_test)[1]])
for k in range(np.shape(X_big_test)[1]):
    for p in range(5):
        if p+1 == X_big_test[0,k]:
            y_one_hot_test[p,k] = 1


# Write processed data into a file.
filename = "Final_Data"
if os.path.exists(filename + ".csv"): #Checks if file exists and if true removes this makes sure it overwrites it.
    print("Removing old "+ filename + ".csv"+ " before writing new.")
    os.remove(filename + ".csv")
np.savetxt(filename + ".csv", data_shuffeled ,delimiter = ';') # Prints data to file
print("Final_Data Length is "+ str(np.shape(data_shuffeled)[1]))

filename_y_one_hot = "Y_One_Hot"
if os.path.exists(filename_y_one_hot + ".csv"): #Checks if file exists and if true removes this makes sure it overwrites it.
    print("Removing old "+ filename_y_one_hot + ".csv"+ " before writing new.")
    os.remove(filename_y_one_hot + ".csv")
np.savetxt(filename_y_one_hot + ".csv", y_one_hot ,delimiter = ';') # Prints data to file

print("Final Test Data Length is "+ str(np.shape(X_big_test)[1]))
filename_test = "Test_Data"
if os.path.exists(filename_test + ".csv"):
    print("Removing old "+ filename_test + ".csv"+ " before writing new.")
    os.remove(filename_test + ".csv")
np.savetxt(filename_test+".csv", X_big_test,delimiter = ';')

filename_y_one_hot_test = "Y_One_Hot_Test"
if os.path.exists(filename_y_one_hot_test + ".csv"): #Checks if file exists and if true removes this makes sure it overwrites it.
    print("Removing old "+ filename_y_one_hot_test + ".csv"+ " before writing new.")
    os.remove(filename_y_one_hot_test + ".csv")
np.savetxt(filename_y_one_hot_test + ".csv", y_one_hot_test ,delimiter = ';') # Prints data to file


filename_test_time_series ="Test_Time_Series"
if os.path.exists(filename_test_time_series + ".csv"):
    print("Removing old "+ filename_test_time_series + ".csv"+ " before writing new.")
    os.remove(filename_test_time_series + ".csv")
np.savetxt(filename_test_time_series +".csv", data_test,delimiter = ';')


filename_left_out_slice = "Left_Out_Data"
if os.path.exists(filename_left_out_slice + ".csv"):
    print("Removing old "+ filename_left_out_slice + ".csv"+ " before writing new.")
    os.remove(filename_left_out_slice + ".csv")
np.savetxt(filename_left_out_slice +".csv", np.transpose(left_out_slice),delimiter = ';')
