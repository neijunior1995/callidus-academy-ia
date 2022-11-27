import tensorflow as tf
import os

class Dataset():
    """Esta classe possui os métodos e atributos necessário para realizar o carregamento do dateset utilizado para treinamento e validação
    *diretory*: é o endereço onde estão armazenadas as pastas de treinamento e validação. Dentro do diretório é necessário que os dados treinamento estejam dentro
    da pasta **train** e que as imagens fiquem separadas por pastas que correspondem as suas classes. Os dados de validação precisam está na pasta **validation** e separados
    da mesma que a na pasta train.
    *batch_size*: é o tamanho dos subconjuntos utilizados para realizar o treinamento.
    *img_size*: é o tamanho da imagem utilizado para realizar as inferências.
    *train_dir*: é o diretório de treinamento
    *validation_dir*: é o diretório de validação
    *train_dataset*: é o dataset de treinamento
    *test_dataset*: é dataset de teste
    *class_name*: é o atributo que armazena os nomes das classes e é igual ao nome das pastas utilizadas para separá-las
    """
    def __init__(self,diretory = "dados", batch_size = 32, img_size = (160,160)):
        self.batch_size = batch_size
        self.img_size = img_size
        print(diretory)
        self.train_dir = os.path.join(diretory, 'train')
        self.validation_dir = os.path.join(diretory, 'validation')
        self.train_dataset      = tf.keras.utils.image_dataset_from_directory(self.train_dir,
                                                            shuffle   = True,
                                                            batch_size= self.batch_size,
                                                            image_size= self.img_size)
        
        self.validation_dataset = tf.keras.utils.image_dataset_from_directory(self.validation_dir,
                                                                 shuffle=True,
                                                                 batch_size=batch_size,
                                                                 image_size=self.img_size)
        self.class_name = self.train_dataset.class_names
        
        val_batches = tf.data.experimental.cardinality(self.validation_dataset)
        self.test_dataset = self.validation_dataset.take(val_batches // 5)
        self.validation_dataset = self.validation_dataset.skip(val_batches // 5)
        AUTOTUNE = tf.data.AUTOTUNE

        self.train_dataset = self.train_dataset.prefetch(buffer_size=AUTOTUNE)
        self.validation_dataset = self.validation_dataset.prefetch(buffer_size=AUTOTUNE)
        self.test_dataset = self.test_dataset.prefetch(buffer_size=AUTOTUNE)
    def vizualize_data(self):
        """Método utilizado para visualizar o tamanho dos datasets"""
        print('Números de batchs de treinamento: %d' % tf.data.experimental.cardinality(self.validation_dataset))
        print('Números de batchs de validação: %d' % tf.data.experimental.cardinality(self.validation_dataset))
        print('Números de batchs de teste: %d' % tf.data.experimental.cardinality(self.test_dataset)) 