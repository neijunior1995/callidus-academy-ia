from Modelos.brain   import Brain
from Modelos.dataset import Dataset
import tensorflow as tf
import pickle

class Classificador():
    """Esta é a classe utilizada para armazenar e implementar o modelo utilizado para o treinamento e classificação de imagens
    *dataset*: Atributo utilizado para armazenar o dataset utilizado treinamento.
    *model*: Atributo que armazena a inteligência utilizada para classificar as imagens
    """
    VGG16        = Brain.VGG16       # Carrega a rede VGG16
    RESNET50V2   = Brain.RESNET50V2  # Carrega o modelo ResNet50V2
    MOBILENET    = Brain.MOBILENET   # Carrega o modelo MobileNet
    MOBILENETV2  = Brain.MOBILENETV2 # Carrega o modelo MobileNetV2
    def __init__(self,pretrained_model = 0, input_size = (160,160,3),diretory = "dados"):
        self.input_size = input_size
        if diretory != None:
            self.dataset = Dataset(img_size = input_size[0:2], diretory = diretory)
            self.model = Brain(input_size = input_size,output_size = len(self.dataset.class_name),class_name=self.dataset.class_name)
            self.class_name = self.dataset.class_name
            self.model.define(pretrained_model)

    def compile(self,optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),loss=tf.keras.losses.sparse_categorical_crossentropy):
        """Compila os parâmetros do modelo pré-treinado"""
        self.model.compile(tf.keras.optimizers.Adam(learning_rate=0.0001),loss=tf.keras.losses.sparse_categorical_crossentropy)    
    def train(self, epochs = 10):
        """Realizar o treinamento da rede"""
        return self.model.train(self.dataset, epochs = epochs)
    def fine_tunning(self,  epochs = 10):
        """Realiza o ajuste fino do parâmetros"""
        return self.model.fine_tunning(self.dataset, epochs = epochs)
    def evaluate(self):
        """Realiza o teste com o dataset de teste do modelo"""
        return self.model.evaluate(self.dataset)
    def predict(self,imagem):
        """Realiza a predição baseada em uma imagem"""
        return self.model.predict(imagem)
    def predict_file(self,caminho):
        """Realiza a predição a partir de uma arquivo"""
        return self.model.predict_file(caminho)
    def save_model(self,caminho="model.json"):
        # serialize weights to HDF5
        self.model.model.save(caminho+'.h5')
        self.rede_parametros = {"class_name": self.class_name,
                                "input_size": self.input_size}
        with open(caminho+'.class', "wb") as f:
            pickle.dump(self.rede_parametros,f)
        print("Modelo salvo no disco")
    def load_model(self,caminho="model"):
        """Carrega um modelo treinado"""
        from keras.models import model_from_json
        # Open the file in binary mode
        with open(caminho+'.class', 'rb') as file:
        # Call load method to deserialze
            self.rede_parametros = pickle.load(file)
        # load weights into new model
        self.input_size = self.rede_parametros["input_size"]
        self.class_name = self.rede_parametros["class_name"]
        self.model = Brain(input_size = self.input_size,output_size = len(self.class_name),class_name=self.class_name)
        self.model.model = tf.keras.models.load_model(caminho+'.h5')
        print(self.model.model)
        print("Modelo carregado no disco")