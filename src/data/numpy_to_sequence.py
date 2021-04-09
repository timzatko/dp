import math

from tensorflow.keras.utils import Sequence


class MRIDummySequence(Sequence):
    def __init__(self, image_x, image_y, batch_size):
        self.image_x = image_x
        self.image_y = image_y
        self.batch_size = batch_size
        
    def size(self):
        return len(self.image_x)

    def __len__(self):
        return math.ceil(len(self.image_x) / self.batch_size)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        return self.image_x[start:end], self.image_y[start:end]
    
    
def numpy_to_sequence(image_x, image_y, batch_size=8):
    return MRIDummySequence(image_x, image_y, batch_size)
