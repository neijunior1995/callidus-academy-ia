import tensorflow as tf
from utils.utils import dataAugmentation, preProcess, globalAverage, prediction_layer
import cv2
import numpy as np

class Brain():
    """Esta classe contém o modelo e os métodos necessários para realizar 
    a inferência de uma rede neural pré-treinada para classificação de imagens.
    """
    VGG16      = 0 # Carrega a rede VGG16
    RESNET50V2 = 1 # Carrega o modelo ResNet50V2
    MOBILENET  = 2 # Carrega o modelo MobileNet
    MOBILENETV2  = 3 # Carrega o modelo MobileNetV2
  
    def __init__(self, output_size , input_size = (160,160,3), class_name=None):
        self.input_size = input_size              # Tamanho dos dados de entrada
        self.class_names = class_name             # Nomes das classes
        self.output_size = output_size            # Tamanho da saída que é igual ao número de classes
        self.model = None;                        # Atributo que armazenará o modelo
        self.base_model = None;
    
    def define(self,pretrained_model=0):
        """ Método da classe responsável por carregar os parâmetros da rede pré-treinada,
    no primeiro carregamento é necessários a conexão com a internet.
    O tipo da rede carregada pode ser selecioadas pelas constantes:
    
    **VGG16**
    **RESNET50V2**
    **MOBILENET**
    **MOBILENETV2**
        """
        if pretrained_model == 0:
            self.base_model = tf.keras.applications.VGG16(input_shape=self.input_size,
                                               include_top=False,
                                               weights='imagenet')
        elif pretrained_model == 1:
            self.base_model = tf.keras.applications.ResNet50V2(input_shape=self.input_size,
                                               include_top=False,
                                               weights='imagenet')
        elif pretrained_model == 2:
            self.base_model = tf.keras.applications.MobileNet(input_shape=self.input_size,
                                               include_top=False,
                                               weights='imagenet')
        elif pretrained_model == 3:
            self.base_model = tf.keras.applications.MobileNetV2(input_shape=self.input_size,
                                               include_top=False,
                                               weights='imagenet')
        inputs = tf.keras.Input(shape=self.input_size)
        x = dataAugmentation()(inputs)
        x = preProcess()(x)
        x = self.base_model(x, training = False)
        x = globalAverage()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(self.output_size)(x)
        self.model = tf.keras.Model(inputs, outputs)
        
    def compile(self,optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),loss=tf.keras.losses.sparse_categorical_crossentropy):
        """Compila o modelo e recebe como parâmetros o otimizador e a loss utiliza"""
        self.model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'])
              
    def train(self,dataset, epochs = 10):
        """Realiza o treinamento do modelo"""
        return self.model.fit(dataset.train_dataset,
                    epochs=epochs,
                    validation_data=dataset.validation_dataset)
                    
    def fine_tunning(self,dataset, epochs = 10):
        """Realiza o ajuste fino com a atualização de todos os parâmetros do modelo"""
        self.base_model.trainable = True
        history = self.model.fit(dataset.train_dataset,
                    epochs=epochs,
                    validation_data=dataset.validation_dataset)
        self.base_model.trainable = False
        return history
        
    def evaluate(self, dataset):
        """Realize o teste do modelo"""
        self.model.evaluate(dataset.test_dataset)
        
    def load_image(self,caminho):
        """Carrega uma imagem"""
        imagem = cv2.imread(caminho)
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        imagem = cv2.resize(imagem,self.input_size[0:2])
        imagem = imagem.reshape(1,self.input_size[0],self.input_size[1],self.input_size[2])
        return imagem
        
    def predict(self,imagem):
        """Utiliza uma imagem no formaro de vetor numpy para realizar a classificação"""
        preds = self.model.predict(imagem)
        classes = []
        #for i in range():
           # classes.append(self.class_names[pred])
        #pred = pred[0]
        preds = np.argmax(preds,axis = -1)
        for pred in preds:
            classes.append(self.class_names[pred])
        #pred = self.class_names[pred]
        return classes
        
    def predict_file(self,caminho):
        """Realiza a predição baseada em um caminho"""
        return self.predict(self.load_image(caminho))