#!/usr/bin/env python3

# #@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/deepdream.ipynb

import time

import tensorflow as tf
import numpy as np
import matplotlib as mpl
import IPython.display as display
import PIL.Image
from tensorflow.keras.preprocessing import image

from deepdream import DeepDream

# source_image can be a URL or local path
# some HTTPS URLs work, some HTTPS URLs don't, but he error is always:
#  urllib.error.URLError: <urlopen error unknown url type: https>
#  ...which could be more helpful.
#source_image = 'https://i2-prod.manchestereveningnews.co.uk/incoming/article10673597.ece/ALTERNATES/s1200b/JS79535762.jpg'
#source_image = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
source_image = '/Users/jon/Downloads/JS79535762.jpg'

"""
The idea in DeepDream is to choose a layer (or layers) and maximize the "loss" in a way that the image 
increasingly "excites" the layers. The complexity of the features incorporated depends on layers chosen 
by you, i.e, lower layers produce strokes or simple patterns, while deeper layers give sophisticated 
features in images, or even whole objects.

The InceptionV3 architecture is quite large (for a graph of the model architecture see TensorFlow's 
research repo). For DeepDream, the layers of interest are those where the convolutions are concatenated. 
There are 11 of these layers in InceptionV3, named 'mixed0' though 'mixed10'. Using different layers will 
result in different dream-like images. Deeper layers respond to higher-level features (such as eyes and 
faces), while earlier layers respond to simpler features (such as edges, shapes, and textures). 
Feel free to experiment with the layers selected below, but keep in mind that deeper layers 
(those with a higher index) will take longer to train on since the gradient computation is deeper.
"""
names = ['mixed2', 'mixed2']
max_image_dimension = 1024
upper_layer_range = 7
batch_process = True

# SOURCE IMAGE LOAD
# get_source_image
#  an image and read it into a NumPy array.
def get_source_image(url_or_path, max_dim=None):
  if url_or_path.startswith('http'):
    name = url_or_path.split('/')[-1]
    image_path = tf.keras.utils.get_file(name, origin=url_or_path)
  else:
    image_path = url_or_path

  img = PIL.Image.open(image_path)
  if max_dim:
    img.thumbnail((max_dim, max_dim))
  return np.array(img)

# Normalize an image
def deprocess(img):
  img = 255*(img + 1.0)/2.0
  return tf.cast(img, tf.uint8)

# Display an image
def show(img):
  stamp = str(int(time.time())) + '_' + ('-'.join(names)) 
  filename = "renders/%s.png" % stamp
  PIL.Image.fromarray(np.array(img)).save(filename)


# Downsizing the image makes it easier to work with.
source_img = get_source_image(source_image, max_dim=max_image_dimension)

# PREPARE FEATURE EXTRACTION MODEL
# Download and prepare a pre-trained image classification model.
base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

# MAIN LOOP
def run_deep_dream_simple(deepdreamInstance, img, steps=100, step_size=0.01):
  # Convert from uint8 to the range expected by the model.
  img = tf.keras.applications.inception_v3.preprocess_input(img)
  img = tf.convert_to_tensor(img)
  step_size = tf.convert_to_tensor(step_size)
  steps_remaining = steps
  step = 0
  while steps_remaining:
    if steps_remaining>100:
      run_steps = tf.constant(100)
    else:
      run_steps = tf.constant(steps_remaining)
    steps_remaining -= run_steps
    step += run_steps

    loss, img = deepdreamInstance(img, run_steps, tf.constant(step_size))
    
    display.clear_output(wait=True)
    show(deprocess(img))
    print ("Step {}, loss {}".format(step, loss))


  result = deprocess(img)
  display.clear_output(wait=True)
  show(result)

  return result

def get_instance_for_model(model):
  return DeepDream(model)

def do_dream():
  layers = [base_model.get_layer(name).output for name in names]
  dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)
  deepdreamInstance = get_instance_for_model(dream_model)
  run_deep_dream_simple(deepdreamInstance, img=source_img, steps=100, step_size=0.01)

# Create the feature extraction model & dream
if batch_process:
  for x in range((upper_layer_range - 1), -1, -1):
    sawtooth_range = (range((upper_layer_range - 1), -1, -1), range(upper_layer_range))[(x % 2) != 0]
    # 9, [9-0] then 8, [0-9] then 7, [9-0] etc.
    for y in sawtooth_range:
      names = ['mixed' + str(x), 'mixed' + str(y)]
      do_dream()

else:
  do_dream()

# ----
print('fin.')