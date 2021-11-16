image_size =260


targetdir1= "./drawings/"
targetdir2= "./hentai/"
targetdir3= "./neutral/"
targetdir4= "./porn/"
targetdir5= "./sexy/"
import time
import os

try:
  os.mkdir(targetdir1)
  os.mkdir(targetdir2)
  os.mkdir(targetdir3)
  os.mkdir(targetdir5)
  os.mkdir(targetdir4)
  
except:
  pass

import webdataset as wds

from webdataset import multi

import cv2
from tqdm import tqdm 
from PIL import Image
import time
import uuid


import itertools

import matplotlib.pylab as plt
import numpy as np
import imageio
import glob          

import time
from detoxify import Detoxify
import multiprocessing
from multiprocessing import Process , Manager


batchsize=1024
SHARDS = "{00000..00002}.tar"

'''
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=8096)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
# Recreate the exact same model, including its weights and the optimizer
#loaded_model = tf.keras.models.load_model('/content/resnetv2-nsfw/nsfweffnetv2-lion.h5',custom_objects={"KerasLayer":hub.KerasLayer})
'''


import webdataset as wds
import torch
import tensorflow as tf

def get_class_string_from_index(index):
  for class_string, class_index in generator.class_indices.items():
      if class_index == index:
        return class_string

def filter_dataset(item): # For e.g. C@H which (rarely) has no caption available.
      if 'txt' not in item:
          return False
      if 'jpg' not in item:
          return False
      return True


def image_classifier(caption_list,prediction_list):


  from tensorflow.python.data.experimental.ops.distribute_options import AutoShardPolicy
  from tensorflow.python.data.ops.dataset_ops import _NumpyIterator as NumpyIterator
  import tensorflow as tf
  import tensorflow_hub as hub

  ds = wds.WebDataset("./laion400m-dat-release/"+SHARDS, handler=wds.ignore_and_continue).select(filter_dataset).decode('rgb').to_tuple('jpg', 'txt')

  dl = wds.WebLoader(ds, shuffle=False, num_workers=16, batch_size=batchsize, prefetch_factor=4*batchsize)  #, prefetch_factor=4*batchsize, pin_memory=True
  c=0
  start =time.time()



  model = tf.keras.models.load_model('nsfweffnetv2-b02-3epochs.h5',custom_objects={"KerasLayer":hub.KerasLayer})
  os.system("nvidia-smi")
  # Show the model architecture
  #model.summary()


  c=0
  start = time.time()

  print("starting loader")
  for im_arr, txt in dl:
    start = time.time()
    c+=1
    im_arr = tf.image.resize(im_arr, [260,260], antialias=True)
    #print (im_arr.shape)
    prediction_scores = model.predict(im_arr)
    prediction_list.append(prediction_scores)
    captions= []
    txt_list = list (txt)
    for e in txt_list:
      captions.append(e[:200])

    '''
    print(type(captions[0]))
    print(captions[0])
    print(captions)
    '''
    caption_list.append(captions)
    print(c)
    print("image predition time")
    print( time.time()-start)
  del model
  tf.keras.backend.clear_session()

#caption_list
#prediction_list

start =time.time()

n_drawings =0
n_hentai =0
n_neutral =0
n_porn =0
n_sexy =0
manager = Manager()
prediction_list= manager.list()
caption_list= manager.list()
p=[]
p.append(Process(target=image_classifier, args=(caption_list,prediction_list )))
p[0].start()
p[0].join()


#reload(tensorflow_hubs)
model_txt = Detoxify('multilingual', device='cuda')
os.system("nvidia-smi")

for i in range(len(caption_list)):
  #start = time.time()
  #print(type(caption_list[i]))

  text_res = model_txt.predict(caption_list[i])

  predicted_indices =[]
  for j in range(len(caption_list[i])):

    predicted_indices.append( np.argmax(prediction_list[i][j]))
    #print(prediction_list[i].shape)
    dist = np.array(tf.nn.softmax(prediction_list[i][j]))
    dist[1]=dist[1]+text_res["sexual_explicit"][j] + text_res["toxicity"][j]  
    dist[3]=dist[3]+text_res["sexual_explicit"][j] + text_res["toxicity"][j]  
    dist[4]=dist[4]+text_res["sexual_explicit"][j] + text_res["toxicity"][j]  

    predicted_index = np.argmax(dist)
    #print("predicted_index")
    #print(predicted_index)
    if predicted_index==0:
      #imageio.imwrite(targetdir1+str(n_drawings+100000000)+".jpg", im_arr[j])  #content/nsfw_data_scraper/data/train/porn/
      n_drawings +=1
      #print("n_drawings: "+str(n_drawings))
    if predicted_index==1:
      #imageio.imwrite(targetdir2+str(n_hentai+100000000)+".jpg", im_arr[j])  #content/nsfw_data_scraper/data/train/porn/
      n_hentai +=1
      #print("n_hentai: "+str(n_hentai))
    if predicted_index==2:
      #imageio.imwrite(targetdir3+str(n_neutral+100000000)+".jpg", im_arr[j])  #content/nsfw_data_scraper/data/train/porn/
      n_neutral +=1
      #print("n_neutral: "+str(n_neutral))
    if predicted_index==3:
      #imageio.imwrite(targetdir4+str(n_porn+100000000)+".jpg", im_arr[j])  #content/nsfw_data_scraper/data/train/porn/
      n_porn +=1
      #print("n_porn: "+str(n_porn))
    if predicted_index==4:
      #imageio.imwrite(targetdir5+str(n_sexy+100000000)+".jpg", im_arr[j])  #content/nsfw_data_scraper/data/train/porn/
      n_sexy +=1
      #print("n_sexy: "+str(n_sexy))
  print(i)
  #print("txt predition time")
  #print( time.time()-start)
  
  #start = time.time()

print("n_drawings: "+str(n_drawings))
print("n_hentai: "+str(n_hentai))
print("n_neutral: "+str(n_neutral))
print("n_porn: "+str(n_porn))
print("n_sexy: "+str(n_sexy))
print( time.time()-start)