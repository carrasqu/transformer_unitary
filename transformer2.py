from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
from POVM import POVM
import itertools as it
import slicetf
from MPS import MPS
import sys
import os



def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
  sines = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  cosines = np.cos(angle_rads[:, 1::2])
  
  pos_encoding = np.concatenate([sines, cosines], axis=-1)
  
  pos_encoding = pos_encoding[np.newaxis, ...]
    
  return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, -1), tf.float32)
  
  # add extra dimensions so that we can add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
  
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    
  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.

  if mask is not None:
    #print(scaled_attention_logits.shape,mask.shape) 
    scaled_attention_logits += (mask * -1e9)  
    
  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
  #print(attention_weights.shape,attention_weights)
  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
  #print("output word 1",output[:,:,0,:])
  return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)
    
    self.dense = tf.keras.layers.Dense(d_model)
    #print(self.depth,"depth")
        
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]
    
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    #print(q.shape,k.shape,v.shape,"shapes??") 
    
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
    #print("scaled_attention w1",scaled_attention[:,:,0,:])   
    #print("scaled_attention w2",scaled_attention[:,:,1,:])
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
    
    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
    #print("outputMHA1",output.shape, "w1", output[:,0,:],"w2", output[:,1,:])
        
    return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)
 
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    #self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    #self.layernorm1 = tf.keras.layers.experimental.LayerNormalization(epsilon=1e-6)
    #self.layernorm2 = tf.keras.layers.experimental.LayerNormalization(epsilon=1e-6)
    #self.layernorm3 = tf.keras.layers.experimental.LayerNormalization(epsilon=1e-6) 
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    #self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)
    
    
  def call(self, x, training, 
           look_ahead_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    #print(attn1.shape,attn1[:,0,:],"attn1 first word")
    #print(attn1.shape,attn1[:,1,:],"attn1 second word") 
    attn1 = self.dropout1(attn1, training=training)
    #print(attn1.shape,attn1[:,0,:],"attn1 first word dropout")
    #print(attn1.shape,attn1[:,1,:],"attn1 second word dropout")
    #out1 = self.layernorm1(attn1 + x)
    out1 = attn1 + x
    s = tf.shape(out1)
    out1 = tf.reshape(out1,[s[0]*s[1],s[2]])
    out1 = self.layernorm1(out1)
    out1 = tf.reshape(out1,[s[0],s[1],s[2]]) 
    #print("out1 w1",out1[:,0,:],"out1 w2",out1[:,1,:]) 
    #attn2, attn_weights_block2 = self.mha2(
    #    enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
    #attn2 = self.dropout2(attn2, training=training)
    #out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
    
    ffn_output = self.ffn(out1)  # (batch_size, target_seq_len, d_model)
    #print("ffn_output word 1",ffn_output[:,0,:])
    #print("ffn_output word 2",ffn_output[:,1,:])  
    ffn_output = self.dropout3(ffn_output, training=training)
    #out3 = self.layernorm3(ffn_output + out1)  # (batch_size, target_seq_len, d_model)
    out3 = ffn_output + out1
    s = tf.shape(out3)
    out3 = tf.reshape(out3,[s[0]*s[1],s[2]])
    out3 = self.layernorm1(out3)
    out3 = tf.reshape(out3,[s[0],s[1],s[2]]) 
    #out3 = self.layernorm3(ffn_output)
     
    return out3, attn_weights_block1


class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, 
               rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    
    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.pos_encoding = positional_encoding(MAX_LENGTH, self.d_model)
    
    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)
    
  def call(self, x, training, 
           look_ahead_mask):

    seq_len = tf.shape(x)[1]
    attention_weights = {}
    
    x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]
    
    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x, block1 = self.dec_layers[i](x,  training,
                                             look_ahead_mask)
      
      attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
      #attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
    
    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights


class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               target_vocab_size, rate=0.1,bias='zeros'):
    super(Transformer, self).__init__()

    #self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
    #                       input_vocab_size, rate)

    self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                           target_vocab_size, rate)

    #self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    bi = tf.constant_initializer(bias)
    self.final_layer = tf.keras.layers.Dense(target_vocab_size,kernel_initializer='zeros',bias_initializer=bi) 
    
  def call(self, tar, training, 
           look_ahead_mask):

    #enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
    
    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, attention_weights = self.decoder(
        tar, training, look_ahead_mask)
    
    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
    
    return final_output, attention_weights



class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def create_masks( tar):
  # Encoder padding mask
  #enc_padding_mask = create_padding_mask(inp)

  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  #dec_padding_mask = create_padding_mask(inp)

  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask(tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

  return combined_mask 




num_layers = 2 #4
d_model = 32 #128 #128
dff = 32 # 128 # 512
num_heads = 4 # 8


batch_size = 1000

target_vocab_size = 4 # number of measurement outcomes
input_vocab_size = target_vocab_size
dropout_rate = 0.0

MAX_LENGTH = 20 # number of qubits

povm_='4Pauli'

povm = POVM(POVM=povm_, Number_qubits=MAX_LENGTH)

mps = MPS(POVM=povm_,Number_qubits=MAX_LENGTH,MPS="Graph")

bias = povm.getinitialbias("+")

EPOCHS = 40

j_init = 0

Ndataset = 2000000 # for training each model

Nbatch_sample = 100000 # size of the batch when I call the sampling

Ndataset_eval = 200000 # # number of samples to evaluate the model at the end 

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, dropout_rate,bias)


learning_rate = CustomSchedule(d_model)

#optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
#                                     epsilon=1e-9)

optimizer = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

def sample(Nsamples=1000):

  #encoder_input = tf.ones([Nsamples,MAX_LENGTH,d_model]) #(inp should be? bsize, sequence_length, d_model)
  output = tf.zeros([Nsamples,1])
  logP = tf.zeros([Nsamples,1])

  for i in range(MAX_LENGTH):
    #print("conditional sampling at site", i)
    combined_mask = create_masks(output)

    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, attention_weights = transformer(output, # self, tar, training,look_ahead_mask
                                                 False,
                                                 None)
    #if i == MAX_LENGTH-1:
    #    logP = tf.math.log(tf.nn.softmax(predictions,axis=2)+1e-10) # to compute the logP of the sampled config after sampling

    predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size) # select  # select the last word from the     seq_len dimension

    predictions = tf.reshape(predictions,[-1,target_vocab_size])  # (batch_size, 1, vocab_size)

    predicted_id = tf.random.categorical(predictions,1) # sample the conditional distribution

    lp = tf.math.log(tf.nn.softmax(predictions,axis=1)+1e-10)

    ohot = tf.reshape(tf.one_hot(predicted_id,target_vocab_size),[-1,target_vocab_size])
    
    preclp = tf.reshape(tf.reduce_sum(ohot*lp,[1]),[-1,1])
   
    logP = logP + preclp

    output = tf.concat([output, tf.cast(predicted_id,dtype=tf.float32)], axis=1)

  output = tf.slice(output, [0, 1], [-1, -1]) # Cut the input of the initial call (zeros)

  oh = tf.one_hot(tf.cast(output,dtype=tf.int64),target_vocab_size) # one hot vector of the sample

  #logP = tf.reduce_sum(logP*oh,[1,2]) # the log probability of the configuration
  #print(logP)
  return output,logP #, attention_weights


def logP(config,training=False):

  Nsamples =  tf.shape(config)[0]
  #encoder_input = tf.ones([Nsamples,MAX_LENGTH,d_model]) #(inp should be? bsize, sequence_length, d_model)
  init  = tf.zeros([Nsamples,1])
  output = tf.concat([init,tf.cast(config,dtype=tf.float32)],axis=1)
  output = output[:,0:MAX_LENGTH]

  combined_mask = create_masks(output)

  # predictions.shape == (batch_size, seq_len, vocab_size) # self, tar, training,look_ahead_mask
  predictions, attention_weights = transformer(output,training,combined_mask)

  # predictions (Nsamples/b_size, MAX_LENGTH,vocab_size)
  # print(predictions)
  logP = tf.math.log(tf.nn.softmax(predictions,axis=2)+1e-10)
  #print(logP[:,0,:],logP.shape,"config+0",output)
  oh = tf.one_hot(config,target_vocab_size)
  logP = tf.reduce_sum(logP*oh,[1,2])

  return logP #, attention_weights


def flip2_tf(S,O,K,site):
    Ns = tf.shape(S)[0]
    N  = tf.shape(S)[1]
    flipped = tf.reshape(tf.keras.backend.repeat(S, K**2),(Ns*K**2,N))
    a = tf.constant(np.array(list(it.product(range(K), repeat = 2)),dtype=np.float32)) # possible combinations of outcomes on 2 qubits
    s0 = flipped[:,site[0]]
    s1 = flipped[:,site[1]]
    a0 = tf.reshape(tf.tile(a[:,0],[Ns]),[-1])
    a1 = tf.reshape(tf.tile(a[:,1],[Ns]),[-1])
    flipped = slicetf.replace_slice_in(flipped)[:,site[0]].with_value(tf.reshape( a0 ,[K**2*Ns,1]))
    flipped = slicetf.replace_slice_in(flipped)[:,site[1]].with_value(tf.reshape( a1 ,[K**2*Ns,1]))
    a = tf.tile(a,[Ns,1])
    indices_ = tf.cast(tf.concat([a,tf.reshape(s0,[tf.shape(s0)[0],1]),tf.reshape(s1,[tf.shape(s1)[0],1])],1),tf.int32)
    ##getting the coefficients of the p-gates that accompany the flipped samples
    Coef = tf.gather_nd(O,indices_)
    # If some coefficients are zero, then eliminate those configurations (could be improved I believe)
    mask = tf.where(np.abs(Coef)<1e-13,False,True)
    Coef = tf.boolean_mask(Coef,mask)
    flipped = tf.boolean_mask(flipped,mask)
    
    ## transform samples to one hot vector
    #flipped = tf.one_hot(tf.cast(flipped,tf.int32),depth=K)
    #flipped = tf.reshape(flipped,[tf.shape(flipped)[0],tf.shape(flipped)[1]*tf.shape(flipped)[2]])
    return flipped,Coef #,indices


def loss_function(flip,co,gtype,Ns):
    #f = tf.cond(tf.equal(gtype,1), lambda: target_vocab_size, lambda: target_vocab_size**2)
    c = tf.cast(flip, dtype=tf.int64)
    lnP = logP(c,training=True)
    #oh =  tf.one_hot(tf.cast(flip,tf.int32),depth=target_vocab_size)
    co = tf.cast(co,dtype = tf.float32)
    loss = -(1.0/tf.cast(Ns,tf.float32))*tf.reduce_sum(co * lnP)
    return loss

@tf.function
def train_step(flip,co,gtype,Ns):

    with tf.GradientTape() as tape:
        loss = loss_function(flip,co,gtype,Ns)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    #print(gradients)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))


    return loss

#sys.exit(0)

if not os.path.exists("samples"):
    os.makedirs("samples")

for j in range(j_init,MAX_LENGTH-1):

    sites=[j,j+1] # on which sites to apply the gate
    gate = povm.p_two_qubit[1] # CZ gate

    
    if Ndataset != 0:
        Ncalls = Ndataset /Nbatch_sample
        samples,lP = sample(Nbatch_sample) # get samples from the model
        lP = np.reshape(lP,[-1,1])

        for k in range(int(Ncalls)):
            sa,llpp = sample(Nbatch_sample)
            samples = np.vstack((samples,sa))
            llpp =np.reshape(llpp,[-1,1])
            lP =  np.vstack((lP,llpp))

    gtype = 2 # 2-qubit gate

    nsteps = int(samples.shape[0] / batch_size)
    bcount = 0
    counter=0
    samples = tf.stop_gradient(samples)

    ept = tf.random.shuffle(samples)


    for epoch in range(EPOCHS):

            print("epoch", epoch,"out of ", EPOCHS,"site", j,flush=True)
            for idx in range(nsteps):

                if bcount*batch_size + batch_size>=Ndataset:
                    bcount=0
                    ept = tf.random.shuffle(samples)

                batch = ept[ bcount*batch_size: bcount*batch_size+batch_size,:]
                bcount=bcount+1


                flip,co = flip2_tf(batch,gate,target_vocab_size,sites)

                Ns = tf.shape(batch)[0]
                l = train_step(flip,co,gtype,Ns)

                #samp,llpp = sample(100000) # get samples from the mode

                #np.savetxt('./samples/samplex_'+str(epoch)+'_iteration_'+str(idx)+'.txt',samp+1,fmt='%i')
                #np.savetxt('./samples/logP_'+str(epoch)+'_iteration_'+str(idx)+'.txt',llpp)                 
                #cFid, cFidError, KL, KLError = mps.cFidelity(tf.cast(samp,dtype=tf.int64),llpp)

                #print(epoch,idx,l)
                #a = (np.array(list(it.product(range(4), repeat = 2)),dtype=np.uint8))
                #l = np.sum(np.exp(logP(a)  ))
                #print("prob",l)

print("training done",flush=True)
#samples,lnP = sample(Ndataset)
if Ndataset_eval != 0:
    Ncalls = Ndataset /Nbatch_sample
    samples,lP = sample(Nbatch_sample) # get samples from the model
    lP = np.reshape(lP,[-1,1])

    for k in range(int(Ncalls)):
        sa,llpp = sample(Nbatch_sample)
        samples = np.vstack((samples,sa))
        llpp =np.reshape(llpp,[-1,1])
        lP =  np.vstack((lP,llpp))

np.savetxt('./samples/samplex.txt',samples+1,fmt='%i')
np.savetxt('./samples/logP.txt',lP)

# classical fidelity
cFid, cFidError, KL, KLError = mps.cFidelity(tf.cast(samples,dtype=tf.int64),lP)
#Fid, FidErrorr = mps.Fidelity(tf.cast(samples,dtype=tf.int64))
#stabilizers,sError = mps.stabilizers_samples(tf.cast(samples,dtype=tf.int64))
print(cFid, cFidError,KL, KLError)
#print(stabilizers,sError,np.mean(stabilizers),np.mean(sError))
