import os
from tensorflow.keras.preprocessing.image import load_img
import tensorflow as tf
import random

def get_paths(dir_name):
  """ Get paths for all files in a given directory.

  Args:
    dir_name (string): directory that you want to get all
                       paths.
  """
  return sorted(
        [
        os.path.join(dir_name, fname)
        for fname in os.listdir(dir_name)
        if fname.endswith(".png")
        ])

class Covid(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths        

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        x = np.zeros((batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)            
            x[j] = img
        y = np.full((batch_size,1), 0)
        for j, path in enumerate(batch_input_img_paths):
          if path[:4] == "CT_N":
            y[j] = 1          
        return x/255., y

def generate_train_test(data_path, batch_size, val_times_batchsize, shuffle=True):
  """ Generates train and test dataset given all data paths, batchsize, and the
  number of times that the validation size is compared to the batch size, this 
  is done because if the validation size and batch size are not multiples we do 
  not validate every sample each pass.

  Args:
    data_path (list): list having paths for all images.
    batch_size (int): batch size for training.
    bal_times_batchsize (int): Number of times that we multiply the batch size
                               to generate the validation dataset.
    shuffle (bool): If you want to shuffle your dataset. (default=True)
  """
  
  if shuffle:
    random.Random(42).shuffle(data_path)
  val_samples = batch_size*val_times_batchsize   
  train_paths = data_path[val_samples:]
  val_paths = data_path[:val_samples]

  return train_paths, val_paths