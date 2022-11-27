from Modelos.brain import Brain
from Modelos.inteligencia import Classificador

model = Classificador()
model.compile()
model.train()
model.fine_tunning()
print(model.predict_file("cat.2038.jpg"))