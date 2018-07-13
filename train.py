import sys
import json
import numpy as np
from tflearn.data_utils import to_categorical
from model import model

def train(fname, out_fname):
    """ 
    All data was stored in a json file.
    """
    # Load dataset
    f = open(fname)
    planesnet = json.load(f)
    f.close()

    # Preprocess image data and labels for input
    X = np.array(planesnet['data']) / 255.
    X = X.reshape([-1,3,20,20]).transpose([0,2,3,1])
    Y = np.array(planesnet['labels'])
    Y = to_categorical(Y, 2)

    # Train the model
    model.fit(X, Y, n_epoch=50, shuffle=True, validation_set=.2,
              show_metric=True, batch_size=128, run_id='planesnet')

    # Save trained model
    model.save(out_fname)


if __name__ == "__main__":
    """
    Usage: 
        python3 train.py [json_path] [model_out_path]
    """
    train(sys.argv[1], sys.argv[2])
