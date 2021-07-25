class AnyModel:
    def __init__(self, model_path):
        self._net = None
        self.model_path = model_path

    def load_model(self):
        from keras.models import load_model as _load
        self._net = _load(self.model_path)

    def save_model(self):
        self._net.save(self.model_path)

    @property
    def net(self):
        return self._net