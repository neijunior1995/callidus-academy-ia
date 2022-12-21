from Modelos.brain import Brain
from Modelos.inteligencia import Classificador
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolo7.pt', help='initial weights path')
model = Classificador()
model.compile()
model.train(epochs = 1)
model.fine_tunning(epochs = 1)
print(model.predict_file("cat.2038.jpg"))