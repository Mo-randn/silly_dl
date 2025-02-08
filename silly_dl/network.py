


class NeuralNetworkTraining:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x, labels):
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                x = layer.forward(x, labels)
            else:
                x = layer.forward(x)
        return x
    
    def backward(self):
        grad_output = 1.0
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output
    
    def update_params(self, learning_rate): 
        for layer in self.layers:
            layer.update_params(learning_rate)

    def parameters(self):
        return [layer.parameters() for layer in self.layers]
    
    def print_parameters_count(self):
        count = 0
        for layer in self.layers:
            pars = layer.parameters()
            layer_count = 0
            for par in pars:
                layer_count += len(par.flatten())
            print(f"Layer parameters: {layer_count}")
            count += layer_count
        return count