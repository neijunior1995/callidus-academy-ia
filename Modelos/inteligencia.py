from Modelos.brain   import Brain
from Modelos.dataset import Dataset
class Classificador():
    VGG16        = Brain.VGG16       # Carrega a rede VGG16
    RESNET50V2   = Brain.RESNET50V2  # Carrega o modelo ResNet50V2
    MOBILENET    = Brain.MOBILENET   # Carrega o modelo MobileNet
    MOBILENETV2  = Brain.MOBILENETV2 # Carrega o modelo MobileNetV2
    def __init__(self,pretrained_model = 0, input_size = (160,160,3)):
        self.dataset = Dataset(img_size = input_size[0:2])
        self.model = Brain(input_size = input_size,output_size = len(self.dataset.class_name),class_name=self.dataset.class_name)
        self.model.define(pretrained_model)
        self.model.compile()
        
    def train(self, epochs = 10):
        return self.model.train(self.dataset, epochs = epochs)
    def fine_tunning(self,  epochs = 10):
        return self.model.fine_tunning(self.dataset, epochs = epochs)
    def evaluate(self):
        return self.model.evaluate(self.dataset)
    def predict(self,imagem):
        return self.model.predict(imagem)
    def predict_file(self,caminho):
        return self.model.predict_file(caminho)
    def save_model(self,caminho="model.json"):
        # serialize model to JSON
        model_json = self.model.model.to_json()
        with open(caminho, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(caminho)
        print("Saved model to disk")
    def load_model(self,caminho="model.json"):
        from keras.models import model_from_json
        # load json and create model
        json_file = open(caminho, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.model = loaded_model.load_weights(caminho)
        print("Loaded model from disk")