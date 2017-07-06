import pickle

import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import pickle
from random import shuffle
from scipy.misc import imread
from scipy.misc import imresize
import tensorflow as tf

from CocoDataGenerator import CocoDataGenerator
from Generator import Generator

from ssd_keras.ssd import SSD300
from ssd_keras.ssd_training import MultiboxLoss
from ssd_keras.ssd_utils import BBoxUtility


base_data_dir = './data/'

data_generator = CocoDataGenerator(['people', 'car'],'D:/coco/annotations/instances_train2014.json',base_data_dir + 'coco/')

cat, data = data_generator.get_data()

print(len(data))


NUM_CLASSES = len(cat) + 1
input_shape = (300, 300, 3)

priors = pickle.load(open(base_data_dir + 'prior_boxes_ssd300.pkl', 'rb'))
bbox_util = BBoxUtility(NUM_CLASSES, priors)


#gt = pickle.load(open('gt_pascal.pkl', 'rb'))
gt = data
keys = sorted(gt.keys())
num_train = int(round(0.8 * len(keys)))
train_keys = keys[:num_train]
val_keys = keys[num_train:]
num_val = len(val_keys)

path_prefix = data_generator.coco_image_dir
gen = Generator(gt, bbox_util, 10, path_prefix,
                train_keys, val_keys,
                (input_shape[0], input_shape[1]), do_crop=False)


model = SSD300(input_shape, num_classes=NUM_CLASSES)
model.load_weights(base_data_dir + '/checkpoints/weights.09-0.95.hdf5', by_name=True)
#
# freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
#           'conv2_1', 'conv2_2', 'pool2',
#           'conv3_1', 'conv3_2', 'conv3_3', 'pool3']#,
# #           'conv4_1', 'conv4_2', 'conv4_3', 'pool4']
#
# for L in model.layers:
#     if L.name in freeze:
#         L.trainable = False

def schedule(epoch, decay=0.9):
    return base_lr * decay**(epoch)

tensorboard_callback = keras.callbacks.TensorBoard(log_dir= base_data_dir + '/logs',
                                                   histogram_freq=0,
                                                   write_graph=True, write_images=True
                                                   )
tensorboard_callback.set_model(model)

callbacks = [keras.callbacks.ModelCheckpoint(base_data_dir + '/checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                             verbose=1,
                                             save_weights_only=True),
             keras.callbacks.LearningRateScheduler(schedule),
             tensorboard_callback
             ]

base_lr = 3e-4
optim = keras.optimizers.Adam(lr=base_lr)
# optim = keras.optimizers.RMSprop(lr=base_lr)
# optim = keras.optimizers.SGD(lr=base_lr, momentum=0.9, decay=decay, nesterov=True)
model.compile(optimizer=optim,
              loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss)

nb_epoch = 30
history = model.fit_generator(gen.generate(True), gen.train_batches,
                              nb_epoch, verbose=1,
                              callbacks=callbacks,
                              validation_data=gen.generate(False),
                              nb_val_samples=gen.val_batches,
                              nb_worker=1)



inputs = []
images = []
img_path = path_prefix + sorted(val_keys)[0]
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(imread(img_path))
inputs.append(img.copy())
inputs = preprocess_input(np.array(inputs))

preds = model.predict(inputs, batch_size=1, verbose=1)
results = bbox_util.detection_out(preds)

reds = model.predict(inputs, batch_size=1, verbose=1)
results = bbox_util.detection_out(preds)

for i, img in enumerate(images):
    # Parse the outputs.
    det_label = results[i][:, 0]
    det_conf = results[i][:, 1]
    det_xmin = results[i][:, 2]
    det_ymin = results[i][:, 3]
    det_xmax = results[i][:, 4]
    det_ymax = results[i][:, 5]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.1]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = plt.cm.hsv(np.linspace(0, 1, 4)).tolist()

    #plt.imshow(img / 255.)
    #currentAxis = plt.gca()

    for i in range(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * img.shape[1]))
        ymin = int(round(top_ymin[i] * img.shape[0]))
        xmax = int(round(top_xmax[i] * img.shape[1]))
        ymax = int(round(top_ymax[i] * img.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
        #         label_name = voc_classes[label - 1]
        display_txt = '{:0.2f}, {}'.format(score, label)
        coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
        color = colors[label]
        cv2.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        cv2.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})

    cv2.imshow("SSD result", img)
    cv2.waitKey(0)
