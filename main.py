import tensorflow as tf
import resnet
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import reuters
import cv2


def resize_image_arr(img_arr):
  x_resized_list = []
  for i in range(img_arr.shape[0]):
    img = img_arr[0]
    resized_img = resize(img, (224, 224))
    x_resized_list.append(resized_img)
  return np.stack(x_resized_list)
  
def main():
  training = True
  
  if training == True:
    datasets = tf.keras.datasets.cifar10

    (train_img, train_y), (test_img, test_y) = datasets.load_data()

    norm_train_img, norm_test_img = train_img / 255.0, test_img / 255.0


    train(norm_train_img, norm_test_img, train_y, test_y)
    
  model = load_model('best_model.h5')
  
  image = cv2.imread('image3.jpg')

  print(image.shape)
  re_image = resize_cifar10(image)
  cv2.imshow('image', image)

  cv2.waitKey(0)
  cv2.destroyAllWindows()


  pred_result = model.predict(re_image)

  best_class = predict_label(pred_result)

  print("이 사진은", best_class)
  
def train(norm_train_img, norm_test_img, train_y, test_y):
  # configuration
  batch = 64
  epoch = 50

  model = resnet.res_net50_model()

  es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)
  mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1)

  hist = model.fit(norm_train_img, train_y, batch_size=batch, epochs=epoch, validation_split=0.2, callbacks=[es, mc])
  
  test_loss, test_acc = model.evaluate(norm_test_img, test_y, verbose=0)
  
  '''
  result = hist.history
  print('Test loss:', test_loss)
  print('Test accuracy:', test_acc)
  
  tr_loss = result['loss']
  accuracy = result['accuracy']
  val_loss = result['val_loss']
  val_accuracy = result['val_accuracy']

  new_plot('Train Loss & Validation Loss', 'epochs', 'Traing loss', tr_loss, val_loss, 'train', 'validation')
  new_plot('Train Accuracy & Validation Accuracy', 'epochs', 'Accuracy', accuracy, val_accuracy, 'train', 'validation')
  '''
