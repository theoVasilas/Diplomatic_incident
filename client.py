from typing import Dict, Tuple
from flwr.common import NDArrays
import tensorflow as tf
import flwr as fl 

model = tf.keras.applications.MobileNetV2((32,32,32), classes =10, weights =None)
model.compile("adam", "sparse_categorocal_crossentropy", metrica=["accuracy"])
(x_train, y_train), (x_test, y_test)= tf.keras.datasets.cifar10.load_data()

model.fit(x_train, y_train, epochs=1, baych_size=32)

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config: Dict[str, bool | bytes | float | int | str]) -> NDArrays[ndarray[Any, dtype[Any]]]:
        return super().get_parameters(config)
    
    def fit(self, parametrs, config):
        model.set_weights(parametrs)
        model.fit(x_train, y_train, epoch=1, batch_size=32)
        return model.get_weights(), len(x_train), {}
    
    def evaluate(self, parameters, config):
        model.se_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy":accuracy}
    
fl.client.start_numpy_client(server_address="127.0.0.0:8888",client=FlowerClient)