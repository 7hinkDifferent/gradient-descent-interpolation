import torch

class Interpolator(object):
    model_dict = {}

    def register(self, name):
        def wrapper(model_class):
            self.model_dict[name] = model_class
            return model_class
        return wrapper

    def build(self, name, *args, **kwargs):
        if name not in self.model_dict:
            raise ValueError("model {} not found".format(name))
        return self.model_dict[name](*args, **kwargs)
    
    def from_pretrained(self, name, path="", freeze=False, *args, **kwargs):
        try:
            model = self.build(name, *args, **kwargs)
            params = torch.load(path, map_location=torch.device("cpu"))
            model.load_state_dict(params)
            print("model weights loaded")
        except FileNotFoundError:
            print("model weights not found")
            print("building new model")
            model = self.build(name, *args, **kwargs)
        model.set_freeze(freeze=freeze)
        
        return model

class ObjectiveFunction(object):
    func_dict = {}

    def register(self, name):
        def wrapper(model_class):
            self.func_dict[name] = model_class
            return model_class
        return wrapper

    def build(self, name, *args, **kwargs):
        if name not in self.func_dict:
            raise ValueError("func {} not found".format(name))
        return self.func_dict[name](*args, **kwargs)

interpolator = Interpolator()
objective_function = ObjectiveFunction()