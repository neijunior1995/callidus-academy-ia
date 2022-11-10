import tensorflow as tf
import os

class Dataset():
    def __init__(self,diretory = "data", batch_size = 32, img_size = (160,160)):
        self.batch_size = batch_size
        self.img_size = img_size
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
        print('Números de batchs de treinamento: %d' % tf.data.experimental.cardinality(self.validation_dataset))
        print('Números de batchs de validação: %d' % tf.data.experimental.cardinality(self.validation_dataset))
        print('Números de batchs de teste: %d' % tf.data.experimental.cardinality(self.test_dataset)) 