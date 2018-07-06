import tensorflow as tf
import scipy.io
from scipy.io import wavfile
import numpy as np
import os

print "Loading the data ..."

# load the data
dirname = 'IRMAS-TrainingData'
data = []
dirs = [x[0] for x in os.walk(dirname)]
for idir, subname in enumerate(dirs[1:5]):
	filenames = os.listdir(subname)
	for ifile in filenames:
		rate, f = wavfile.read(subname + '/' + ifile)		
		data.append(np.append(np.mean(f,axis=1).reshape(1,f.shape[0]),idir))
data = np.array(data)[:,1::2] # downsampling to fs=22050
fs = 22050

print "Separating into training and test sets"

# separate data intro training and test sets
np.random.shuffle(data) # first shuffle everything
data_tr = data[:2000,:]
data_vl = data[2000:2200,:]
data_ts = np.append(data[2200:,:fs],data[6000:,-1]) # only two seconds of data
del data

# separating training set into data and labels
x = np.concatenate((data_tr[:,:fs+10],data_tr[:,fs+10:fs*2+20]),axis = 0)
y = np.concatenate((data_tr[:,-1].reshape(data_tr.shape[0],1),data_tr[:,-1].reshape(data_tr.shape[0],1)),axis=0)
del data_tr

# separating validation set into data and labels
x_vl = data_vl[:,:fs+10]
y_vl = data_vl[:,-1].reshape(data_vl.shape[0],1)
del data_vl


print "Building the TF graph"

# data parameters
N = x.shape[0] # total number of training datapoints
D = x.shape[1] # dimensionality
print D
C = 4 # number of classes
C1D = 1024 # filter size
NC1 = 16 # number of channels
C2D = 512 # filter size
NC2 = 16 # number of channels
NH = 256 # hidden units (before softmax)
lr = 0.0001

# make the labels be one-hot vectors
temp = np.zeros((y.shape[0],C))
temp[np.arange(y.shape[0]),y.astype(int)[:,0]-1] = 1
y = temp
temp = np.zeros((y_vl.shape[0],C))
temp[np.arange(y_vl.shape[0]),y_vl.astype(int)[:,0]-1] = 1
y_vl = temp
del temp

# tensorflow placeholders and variables
X = tf.placeholder("float", [None,D,1,1])
Y = tf.placeholder("float", [None, C])
W1 = tf.Variable(tf.truncated_normal([C1D,1,1,NC1], stddev=0.001))
b1 = tf.Variable(tf.truncated_normal([NC1]))
W2 = tf.Variable(tf.truncated_normal([C2D,1,NC1,NC2], stddev=0.001))
b2 = tf.Variable(tf.truncated_normal([NC2]))
W3 = tf.Variable(tf.truncated_normal([(D/4)*NC2,NH], stddev=0.001))                 
b3 = tf.Variable(tf.truncated_normal([NH]))                 
W4 = tf.Variable(tf.truncated_normal([NH,C], stddev=0.001))                 
b4 = tf.Variable(tf.truncated_normal([C]))        

#### Forward pass ###

# Convolution 1                 
C1_out = tf.nn.conv2d(X, W1, [1,1,1,1], padding='SAME')                 
C1_out += b1
C1_out = tf.nn.relu(C1_out)                 

print C1_out

# Max Pooling 1
C1_out_mp = tf.nn.max_pool(C1_out, ksize = [1,2,1,1], strides=[1,2,1,1], padding='SAME')                 

print C1_out_mp
                 
# Convolution 2                 
C2_out = tf.nn.conv2d(C1_out_mp, W2, [1,1,1,1], padding='SAME')                                  
C2_out += b2
C2_out = tf.nn.relu(C2_out)                 

print C2_out

# Max Pooling 2
C2_out_mp = tf.nn.max_pool(C2_out, ksize = [1,2,1,1], strides = [1,2,1,1], padding='SAME')                 

print C2_out_mp

# # Fully connected 1
C2_out_mp = tf.reshape(C2_out_mp,[-1, (D/4)*NC2])                 
print C2_out_mp
H1 = tf.matmul(C2_out_mp, W3) + b3
H1 = tf.nn.relu(H1)
H1 = tf.nn.dropout(H1,0.90) # dropout
                 
# # Fully connected 2 (softmax)                 
scores = tf.matmul(H1, W4) + b4
y_hat = tf.nn.softmax(scores)

# # compute the loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=Y))                  

# # training rule
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
GD_step = optimizer.minimize(loss)

# run the session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print "Initiating Training ..."

nepochs = 100
nminibatch = 10
minibatchsize = int(N/nminibatch)
for i in xrange(nepochs):            
	
	# concatenate data and labels
	data_tr = np.concatenate((x,y),axis=1)
	np.random.shuffle(data_tr) # shuffle the data
	# separate data and labels
	x = data_tr[:,:fs+10]
	y = data_tr[:,fs+10:]
	
	del data_tr

	for iminibatch in xrange(nminibatch):

		training_loss = sess.run(loss, feed_dict={X: x[iminibatch*minibatchsize:iminibatch*minibatchsize + minibatchsize,:].reshape([minibatchsize,x.shape[1],1,1]), Y: y[iminibatch*minibatchsize:iminibatch*minibatchsize + minibatchsize,:]})
		print "Epoch: ", i, ". Minibatch No. ", iminibatch, " of ",nminibatch, " total. Training loss: ", training_loss		
		# gradient descent    
		sess.run(GD_step, feed_dict={X: x[iminibatch*minibatchsize:iminibatch*minibatchsize + minibatchsize].reshape([minibatchsize,x.shape[1],1,1]), Y: y[iminibatch*minibatchsize:iminibatch*minibatchsize + minibatchsize,:]})
	
	training_scores = sess.run(scores, feed_dict={X: x[0:minibatchsize,:].reshape([minibatchsize,x.shape[1],1,1]), Y: y})
	print "Epoch ", i, " finished."	
	print "Training accuracy: " , 100.0*np.sum(np.equal(np.argmax(training_scores,axis=1),np.argmax(y[0:minibatchsize],axis=1)))/minibatchsize

	# validation data assessment	
	validation_scores = sess.run(scores, feed_dict={X: x_vl.reshape([x_vl.shape[0],x_vl.shape[1],1,1]), Y: y_vl})
	print "Validation accuracy: " , 100.0*np.sum(np.equal(np.argmax(validation_scores,axis=1),np.argmax(y_vl,axis=1)))/y_vl.shape[0]	