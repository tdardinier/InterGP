import predictor


class Predictor(predictor.Predictor):

    def __init__(self, n=4, m=1):
        super().__init__(n, m)
        self.name = "identity"

    def predict(self, xx, uu):
        return 0 * xx
