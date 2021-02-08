"""
Created on Feb 4, 2021

"""
import os
import tensorflow as tf
import tensorflow.keras as keras
from utility.parser import parse_args
from utility.load_data import *
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

args = parse_args()
data_generator = Data(path=args.data_path + args.dataset,
                        batch_size=args.batch_size)

# Define initial layer 
class InitialLayer(tf.keras.layers.Layer):
    def __init__(self, n_users, n_items, output_dim):
        super(InitialLayer, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.output_dim = output_dim

    def build(self, input_shape):
        self.user_embedding = self.add_weight(
            name='user_embedding',
            shape=(self.n_users, self.output_dim),
            initializer=tf.random_normal_initializer(stddev=0.1),
            trainable=True,
            dtype=tf.float32
        )
        self.item_embedding = self.add_weight(
            name='item_embedding',
            shape=(self.n_items, self.output_dim),
            initializer=tf.random_normal_initializer(stddev=0.1),
            trainable=True,
            dtype=tf.float32
        )

    def call(self, inputs):
        return tf.concat(self.user_embedding, self.item_embedding)


# Define the rest of the layers
class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, n_users, n_items, norm_adj_mat):
        super(GCNLayer, self).__init__()
        self.norm_adj_mat = norm_adj_mat
        self.n_users = n_users
        self.n_items = n_items
    

    def call(self, inputs):
        #all_embs = tf.concat(inputs[0], inputs[1])
        #new_embedding = tf.linalg.matmul(all_embs, self.norm_adj_mat)
        new_embedding = tf.linalg.matmul(inputs, self.norm_adj_mat)
        new_user_embedding, new_item_embedding = tf.split(new_embedding, [self.n_users, self.n_items], 0)
        #return new_user_embedding, new_item_embedding
        return new_embedding

# Define the model
class KerasLightGCN(tf.keras.Model):
    def __init__(self):
        super(KerasLightGCN, self).__init__()
        self.n_users = data_generator.n_users
        self.n_items = data_generator.n_items
        self.latent_dim = args.embed_size
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.node_dropout = args.node_dropout

        # norm_adj_mat is Ã in the paper
        self.adj_mat, self.norm_adj_mat, self.mean_adj_mat, self.pre_adj_mat = data_generator.get_adj_mat()

        self.layer0 = InitialLayer(self.n_users, self.n_items, self.latent_dim)
        self.gcnlayer = GCNLayer(self.n_users, self.n_items, self.norm_adj_mat)


    def call(self, inputs):
        # Compute embeddings
        initial = self.layer0(None)
        first = self.gcnlayer(initial)
        second = self.gcnlayer(first)
        third = self.gcnlayer(second)
        
        # forward (mul and sum)
        return third


# TODO: Lav custom dropout layer der dropper både nodes og messages

model = KerasLightGCN()

def bpr_loss_func(y_true, y_pred):
    users, pos, neg = data_generator.sample()
    print("im here")



model.compile(
    optimizer='adam',
    loss='bpr_loss_func',
)

train_data = data_generator.train_items


users = tf.data.Dataset.from_generator(train_data)

model.fit(x=users, epochs=2, verbose=1)

