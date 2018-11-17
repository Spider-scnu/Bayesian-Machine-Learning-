import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt


def batch_iter(x, batch_size=64):
    """生成批次数据"""
    data_len, _ = x.shape
    num_batch = int((data_len - 1) / batch_size) + 1
    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    #y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id]

def xavier_init(fan_in, fan_out, constant=1):
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

class CNP(object):
    def __init__(self, network_architecture,
                 transfer_fct=tf.nn.softplus,
                learning_rate = 0.001,
                batch_size = 100):
            
            self.network_architecture = network_architecture
            self.transfer_fct = transfer_fct
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            #self.hidden_recog_size = hidden_recog_size
            #self.hidden_gener_size = hidden_gener_size
            self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])
            self._create_network()
            self._create_loss_optimizer()
            #self.hidden_recog_neural = hidden_recog_neural
            #self.hidden_gener_neural = hidden_gener_neural
            #self.hidden_layer_size = 
            init = tf.global_variables_initializer()
            self.sess = tf.InteractiveSession()
            self.sess.run(init)
    def _create_network(self):
        # Initialize autoencode network weights and biases
        network_weights = self._initialize_weights(**self.network_architecture)
        # Use recognition network to determine mean and 
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq =\
            self._recognition_network(network_weights["weights_recog"], 
                                     network_weights["biases_recog"])
        # Draw one sample z from Gaussian distribution    
        n_z = self.network_architecture["n_z"] # dimension
        eps = tf.random_normal((self.batch_size,n_z), 0, 1, dtype=tf.float32)
        
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
        
        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = \
            self._generator_network(network_weights["weights_gener"],
                                   network_weights["biases_gener"])
    def _initialize_weights(self, hidden_recog_size, hidden_gener_size,
                            hidden_recog_neural, hidden_gener_neural, n_input, n_z):
        all_weights = dict()
        all_weights["weights_recog"] = dict()
        for i in range(hidden_recog_size):
            if i==0:
                string = "h"
                string = "h" + str(i+1)
                all_weights["weights_recog"][string] = tf.Variable(xavier_init(n_input,
                                                                               hidden_recog_neural))
            else:
                string = "h"
                string = "h" + str(i+1)
                all_weights["weights_recog"][string] = tf.Variable(xavier_init(hidden_recog_neural,
                                                                               hidden_recog_neural))
        all_weights["weights_recog"]["out_mean"] = tf.Variable(xavier_init(hidden_recog_neural, n_z))
        all_weights["weights_recog"]["out_log_sigma"] = tf.Variable(xavier_init(hidden_recog_neural, n_z))
        #all_weights["weights_recog"] = {
        #    'h1' : tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
        #    'h2' : tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
        #    'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
        #    'out_log_sigma' : tf.Variable(xavier_init(n_hidden_recog_2, n_z))
        #}
        all_weights["biases_recog"] = dict()
        for i in range(hidden_recog_size):
            if i == 0:
                string = "b"
                string = "b" + str(i+1)
                all_weights["biases_recog"][string] = tf.Variable(tf.zeros([hidden_recog_neural], dtype = tf.float32))
            else:
                string = "b"
                string = "b" + str(i+1)
                all_weights["biases_recog"][string] = tf.Variable(tf.zeros([hidden_recog_neural], dtype = tf.float32))
        all_weights["biases_recog"]["out_mean"] = tf.Variable(tf.zeros([n_z], dtype = tf.float32))
        all_weights["biases_recog"]["out_log_sigma"] = tf.Variable(tf.zeros([n_z], dtype = tf.float32))
        #all_weights["biases_recog"] = {
        #    'b1' : tf.Variable(tf.zeros([n_hidden_recog_1],dtype=tf.float32)),
        #    'b2' : tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
        #    'out_mean' : tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
        #    'out_log_sigma' : tf.Variable(tf.zeros([n_z], dtype = tf.float32))
        #}
        all_weights["weights_gener"] = dict()
        for i in range(hidden_gener_size):
            if i==0:
                #string = "h"
                string = "h" + str(i+1)
                all_weights["weights_gener"][string] = tf.Variable(xavier_init(n_z, hidden_gener_neural))
            else:
                #string = "h"
                string = "h" + str(i+1)
                all_weights["weights_gener"][string] = tf.Variable(xavier_init(hidden_gener_neural, hidden_gener_neural))
        all_weights["weights_gener"]["out_mean"] = tf.Variable(xavier_init(hidden_gener_neural, n_input))
        all_weights["weights_gener"]["out_log_sigma"] = tf.Variable(xavier_init(hidden_gener_neural, n_input))
        
        #all_weights['weights_gener'] = {
        #    'h1' : tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
        #    'h2' : tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
        #    'out_mean' : tf.Variable(xavier_init(n_hidden_gener_2, n_input)),
        #    'out_log_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input))
        #}
        all_weights["biases_gener"] = dict()
        for i in range(hidden_gener_size):
            if i == 0:
                #string = "b"
                string = "b" + str(i+1)
                all_weights["biases_gener"][string] = tf.Variable(tf.zeros([hidden_gener_neural], dtype = tf.float32))
            else:
                #string = "b"
                string = "b" + str(i+1)
                all_weights["biases_gener"][string] = tf.Variable(tf.zeros([hidden_gener_neural], dtype = tf.float32))
        all_weights["biases_gener"]["out_mean"] = tf.Variable(tf.zeros([n_input], dtype = tf.float32))
        all_weights["biases_gener"]["out_log_sigma"] = tf.Variable(tf.zeros([n_input], dtype = tf.float32))
        
        #all_weights['biases_gener'] = {
        #   'b1' : tf.Variable(tf.zeros([n_hidden_gener_1], dtype = tf.float32)),
        #    'b2' : tf.Variable(tf.zeros([n_hidden_gener_2], dtype = tf.float32)),
        #    'out_mean' : tf.Variable(tf.zeros([n_input], dtype = tf.float32)),
        #    'out_log_sigma' : tf.Variable(tf.zeros([n_input], dtype = tf.float32))
        #}
        return all_weights
    
    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        layer = []
        for i in range(self.network_architecture['hidden_recog_size']):
            if i==0:
                string_h = "h" + str(i+1)
                string_b = "b" + str(i+1)
                layer.append(self.transfer_fct(tf.add(tf.matmul(self.x, weights[string_h]), biases[string_b])))
            else:
                string_h = "h" + str(i+1)
                string_b = "b" + str(i+1)
                layer.append(self.transfer_fct(tf.add(tf.matmul(layer[i-1], weights[string_h]), biases[string_b])))
        z_mean = tf.add(tf.matmul(layer[-1], weights['out_mean']), biases['out_mean'])   
        z_log_sigma_sq = \
            tf.add(tf.matmul(layer[-1], weights['out_log_sigma']),biases['out_log_sigma'])
        #layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']),biases['b1']))
        #layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        #z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']), biases['out_mean'])
        #z_log_sigma_sq = \
        #    tf.add(tf.matmul(layer_2, weights['out_log_sigma']),biases['out_log_sigma'])
        return (z_mean, z_log_sigma_sq)
    def _generator_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer = []
        for i in range(self.network_architecture['hidden_gener_size']):
            if i==0:
                string_h = "h" + str(i+1)
                string_b = "b" + str(i+1)
                layer.append(self.transfer_fct(tf.add(tf.matmul(self.z, weights[string_h]), biases[string_b])))
            else:
                string_h = "h" + str(i+1)
                string_b = "b" + str(i+1)
                layer.append(self.transfer_fct(tf.add(tf.matmul(layer[i-1], weights[string_h]), biases[string_b])))
        #layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']), biases['b1']))
        #layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        x_reconstr_mean = \
            tf.nn.sigmoid(tf.add(tf.matmul(layer[-1], weights['out_mean']), biases['out_mean']))
        return x_reconstr_mean
    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution 
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluation of log(0.0)
        reconstr_loss = \
            -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean) +\
                           (1 - self.x)* tf.log(1e-10 + 1 - self.x_reconstr_mean),1)
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence 
        #     between the distribution in latent space induced by the encoder on 
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        latent_loss = -0.5 * tf.reduce_mean(1 + self.z_log_sigma_sq -\
                                            tf.square(self.z_mean) -\
                                            tf.exp(self.z_log_sigma_sq),1) 
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)
        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.cost)
    def partial_fit(self, x):
        """Train model based on mini-batch of input data.
        
        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost), feed_dict = {self.x : x})
        return cost
    def transform(self, x):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict = {self.x:x})
    
    def generate(self, z_mu = None):
        """ Generate data by sampling from latent space.
        
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent 
        space.        
        """
        if z_mu is None:
            z_mu = np.random.normal(size = self.network_architecture['n_z'])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution    
        return self.sess.run(self.x_reconstr_mean, feed_dict = {self.z: z_mu})
    def predict(self, x):
        return self.sess.run(self.x_reconstr_mean, feed_dict = {self.x: x})
    
    
    def fit(self, x, training_epochs = 10, display_step = 5):
        #vae = VariationalAutoencoder(self.network_architecture, learning_rate = learning_rate, batch_size = batch_size)
        n_samples, _ = x.shape
        for epoch in range(training_epochs):
            avg_cost = 0.
            #total_batch = int(n_samples / self.batch_size)
            for batch_xs in batch_iter(x, self.batch_size):
                cost = self.partial_fit(batch_xs)
                avg_cost += cost / n_samples * self.batch_size
            if epoch % display_step == 0:
                print("Epoch: ", '%04d' % (epoch+1), "cost= ", "{:.9f}".format(avg_cost))
            
        #return self
        
        
        