import tensorflow as tf
import keras.applications
import pickle


def dataAugmentation(flip = 'horizontal', rotation = 0.2):
    """Função que retorna um layer utilizado no treinamento que realiza transformações de rotação
e espelhamento para diminuir o overfiting da rede"""
    return tf.keras.Sequential([
    tf.keras.layers.RandomFlip(flip),
    tf.keras.layers.RandomRotation(rotation),
    ])

def preProcess():
    """ Função que retorna o nomalizador"""
    return tf.keras.layers.Rescaling(1./127.5, offset=-1)
def globalAverage():
    return tf.keras.layers.GlobalAveragePooling2D()
    
def prediction_layer(num_class):
    """Método retorna o layer de saída com para o número de classes definido """
    return tf.keras.layers.Dense(num_class, activation='softmax')

def save_model(model, name = 'model.net'):
    with open(name, 'wb') as f:
        pickle.dump(model, f)