#%%
import tensorflow as tf
import json
import keras 
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from keras.applications import VGG16
from keras.models import load_model

from matplotlib import pyplot as plt
import albumentations as alb
from albumentations import (Compose, RandomBrightnessContrast, ImageCompression, 
HueSaturationValue, RandomContrast, HorizontalFlip, Rotate, RandomCrop, VerticalFlip, RGBShift,
RandomGamma)

import matplotlib
import cv2
import uuid
import os
import time
import numpy as np


matplotlib.use('TkAgg')
%matplotlib inline

IMAGES_PATH = os.path.join('data', 'images')
number_of_imgs = 90

if not os.path.exists(IMAGES_PATH):
    os.makedirs(IMAGES_PATH)

#%% Capture images for model 
cap = cv2.VideoCapture(0)   

for imgnum in range(number_of_imgs):

    if len(os.listdir(IMAGES_PATH)) == number_of_imgs:
        print('Max numebr of images reached - Please remove before collecting')
        break 

    else:

        if imgnum == 0:
            print('Image cap not reached - Proceed to collect')

        print('Collecting image {}'.format(imgnum))
        ret, frame = cap.read()
        imgname = os.path.join(IMAGES_PATH,f'{(str(uuid.uuid1()))}.jpg')
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame', frame)
        time.sleep(0.25)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

#%% Pullup Labelme to annotate the images
# !labelme


# limit GPU memory growth to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU)')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# load image into TF data pipeline
images = tf.data.Dataset.list_files('data\\images\\*.jpg', shuffle=False)

def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

images = images.map(load_image)
images.as_numpy_iterator().next()


# %%
image_generator = images.batch(4).as_numpy_iterator()
# %%
plot_images = image_generator.next()

fig, ax = plt.subplots(ncols=4, figsize=(20,5))
for idx, image in enumerate(plot_images):
    ax[idx].imshow(image)
plt.show()

# %% Partition unaugmented data
# for folder in ['train','test','val']:
#     for file in os.listdir(os.path.join('data',folder,'images')):
        
#         filename = file.split('.')[0]+'.json'
#         existing_filepath = os.path.join('data','labels',filename)
#         if os.path.exists(existing_filepath):
#             new_filepath = os.path.join('data',folder,'labels',filename)
#             os.replace(existing_filepath, new_filepath)

# %% Augmentation pipeline setup
transforms = Compose([
            Rotate(limit=40),
            RandomBrightnessContrast(p=0.2),
            RandomCrop(width=450, height=450),
            ImageCompression(quality_lower=85, quality_upper=100, p=0.5),
            HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RGBShift(p=0.2),
            RandomGamma(p=0.2)], 
            bbox_params=alb.BboxParams(format='albumentations',
                                        label_fields=['class_labels']))

def aug_fn(image, img_size):
    data = {"image":image}
    aug_data = transforms(**data)
    aug_img = aug_data["image"]
    aug_img = tf.cast(aug_img/255.0, tf.float32)
    aug_img = tf.image.resize(aug_img, size=[img_size, img_size])
    return aug_img

def process_data(image, label, img_size):
    aug_img = tf.numpy_function(func=aug_fn, inp=[image, img_size], Tout=tf.float32)
    return aug_img, label

#Load test img and annotation 
check_img = cv2.imread(os.path.join('data','train','images','b8cae046-3fa7-11ed-947a-3003c854df3c.jpg'))
with open(os.path.join('data', 'train', 'labels', 'b8cae046-3fa7-11ed-947a-3003c854df3c.json'), 'r') as f:
    label = json.load(f)

label['shapes'][0]['points']

#Extract coordinates and rescale
coords = [0,0,0,0]
coords[0] = label['shapes'][0]['points'][0][0]
coords[1] = label['shapes'][0]['points'][0][1]
coords[2] = label['shapes'][0]['points'][1][0]
coords[3] = label['shapes'][0]['points'][1][1]

coords
coords = list(np.divide(coords,[640,480,640,480]))

# #Apply augumentations and view result
augmented = transforms(image=check_img, bboxes=[coords], class_labels=['face'])
augmented['bboxes'][0][2:]
augmented['bboxes']

cv2.rectangle(augmented['image'],
                tuple(np.multiply(augmented['bboxes'][0][:2], [450,450]).astype(int)),
                tuple(np.multiply(augmented['bboxes'][0][2:], [450,450]).astype(int)),
                    (255,0,0),2)

plt.imshow(augmented['image'])
plt.show()
# %% augumentation pipeline for all images

for partition in ['train','test','val']:

    if len(os.listdir(os.path.join('aug_data',partition,'images'))) > 0:
        print('Augmentation already exists')

        break
    
    for img in os.listdir(os.path.join('data',partition,'images')):
        
        if os.path.exists(img):
            raise Exception 

        img_cv = cv2.imread(os.path.join('data', partition, 'images',img))

        coords = [0,0,0.00001,0.00001]
        label_path = os.path.join('data', partition, 'labels',f'{img.split(".")[0]}.json')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)
            
            coords[0] = label['shapes'][0]['points'][0][0]
            coords[1] = label['shapes'][0]['points'][0][1]
            coords[2] = label['shapes'][0]['points'][1][0]
            coords[3] = label['shapes'][0]['points'][1][1]
            coords = list(np.divide(coords,[640,480,640,480]))

        try:
            for x in range(100):
                augmented = transforms(image=img_cv, bboxes=[coords], class_labels=['face'])
                cv2.imwrite(os.path.join('aug_data', partition, 'images', f'{img.split(".")[0]}.{x}.jpg'), augmented['image'])

                annotation = {}
                annotation['image'] = img

                if os.path.exists(label_path):
                    if len(augmented['bboxes']) == 0:
                        annotation['bboxes'] = [0,0,0,0]
                        annotation['class'] = 0
                    else:
                        annotation['bboxes'] = augmented['bboxes'][0]
                        annotation['class'] = 1
                else:
                    annotation['bboxes'] = [0,0,0,0]
                    annotation['class'] = 0

                with open(os.path.join('aug_data',partition, 'labels', f'{img.split(".")[0]}.{x}.json'),'w') as f:
                    json.dump(annotation, f)
        
        except Exception as e:
            print(e)
# %% Load augmented Images to TF dataset

for augfold in ['train', 'val', 'test']:
    dsimages = tf.data.Dataset.list_files('aug_data\\{}\\images\\*.jpg'.format(augfold), shuffle=False)
    if augfold == 'train':
        train_images = dsimages.map(load_image)
        train_images = train_images.map(lambda x: tf.image.resize(x, (120,120)))
        train_images = train_images.map(lambda x: x/255)
    elif augfold == 'test':
        test_images = dsimages.map(load_image)
        test_images = test_images.map(lambda x: tf.image.resize(x, (120,120)))
        test_images = test_images.map(lambda x: x/255)
    else:
        val_images = dsimages.map(load_image)
        val_images = val_images.map(lambda x: tf.image.resize(x, (120,120)))
        val_images = val_images.map(lambda x: x/255)

# %% Label prep
def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding = "utf-8") as f:
        label = json.load(f)
    
    return [label['class']], label['bboxes']

for laugfold in ['train', 'val', 'test']:
    dsimages_l = tf.data.Dataset.list_files('aug_data\\{}\\labels\\*.json'.format(laugfold), shuffle=False)
    if laugfold == 'train':
        train_labels = dsimages_l.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
    elif laugfold == 'test':
        test_labels = dsimages_l.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
    else:
        val_labels = dsimages_l.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

# %% check partition lengths
count_dict = {}
for e in [len(train_images),len(train_labels),len(test_images),len(test_labels),len(val_images),len(val_labels)]:
    if e == len(train_images):
            count_dict['train'] = e
    elif (e == len(test_images) and e == len(val_images)):
        count_dict['test'] = e
        count_dict['val'] = e
    elif e == len(test_images):
        count_dict['test'] = e
    elif e == len(val_images):
        count_dict['val'] = e 
    else:
        raise Exception
    

# %% Zip label and images together to pair up the numpy arrays this creates final dataset used to train the model
data_batch = 8
data_prefetch = 4

train = tf.data.Dataset.zip((train_images,train_labels))
train = train.shuffle(count_dict['train'])
train = train.batch(data_batch)
train = train.prefetch(data_prefetch)

test = tf.data.Dataset.zip((test_images,test_labels))
test = test.shuffle(count_dict['test'])
test = test.batch(data_batch)
test = test.prefetch(data_prefetch)

val = tf.data.Dataset.zip((val_images,val_labels))
val = val.shuffle(count_dict['test'])
val = val.batch(data_batch)
val = val.prefetch(data_prefetch)

# %% view combined images and labels
data_samples = train.as_numpy_iterator()
res = data_samples.next()
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx in range(4):
    sample_image = res[0][idx]
    sample_coords = res[1][1][idx]

    cv2.rectangle(sample_image,
                tuple(np.multiply(sample_coords[:2], [120,120]).astype(int)),
                tuple(np.multiply(sample_coords[2:], [120,120]).astype(int)),
                    (255,0,0),2)

    ax[idx].imshow(sample_image)

# %% Build deep learning layers (referenced from SSD neural networks * can leverage past work to reference models)
def build_model():
    input_layer = Input(shape=(120,120,3))

    # include_top=False is for removing vgg16 pre-made classification and softmax layer since we want our own
    vgg = VGG16(include_top=False)(input_layer)
    
    # Classification Model
    f1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048, activation='relu')(f1)
    class2 = Dense(1, activation='sigmoid')(class1)

    # Bounding box model
    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(2048, activation='relu')(f2)
    regress2 = Dense(4, activation='sigmoid')(regress1)

    facetracker = Model(inputs=input_layer, outputs=[class2, regress2])
    return facetracker


# %% verify model
facetracker = build_model()
X, y = train.as_numpy_iterator().next()
X.shape

# %% Train model
classes, coords = facetracker.predict(X)
classes, coords

# %% custom losses/optimizers

# localization loss function used in YOLO
batches_per_epoch = len(train)
lr_decay = (1./0.75-1)/batches_per_epoch
opt = tf.keras.optimizers.Adam(learning_rate=0.000095, decay=lr_decay)

def localization_loss(y_true, yhat):
    delta_coord = tf.reduce_sum(tf.square(y_true[:,:2]-yhat[:,:2]))

    h_true = y_true[:,3] - y_true[:,1]
    w_true = y_true[:,2] - y_true[:,0]

    h_pred = yhat[:,3] - yhat[:,1]
    w_pred = yhat[:,2] - yhat[:,0]

    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))

    return delta_coord + delta_size

# Binary cross entropy for classification 
classloss = tf.keras.losses.BinaryCrossentropy()
regressloss = localization_loss

# %% Loss metric test
# classloss(y[0], classes).numpy()
# regressloss(y[1], coords).numpy()
# %% Train NN with custom model class

class FaceTracker(Model): 
    def __init__(self, eyetracker,  **kwargs): 
        super().__init__(**kwargs)
        self.model = eyetracker

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt
    
    def train_step(self, batch, **kwargs): 
        
        X, y = batch
        
        with tf.GradientTape() as tape: 
            classes, coords = self.model(X, training=True)
            
            batch_classloss = self.closs(y[0], classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
            
            total_loss = batch_localizationloss+0.5*batch_classloss
            
            grad = tape.gradient(total_loss, self.model.trainable_variables)
        
        opt.apply_gradients(zip(grad, self.model.trainable_variables))
        
        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}
    
    def test_step(self, batch, **kwargs): 
        X, y = batch
        
        classes, coords = self.model(X, training=False)
        
        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss+0.5*batch_classloss
        
        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}
        
    def call(self, X, **kwargs): 
        return self.model(X, **kwargs)

model = FaceTracker(facetracker)
model.compile(opt, classloss, regressloss)

# %% Training sequence
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback])
hist.history
# %% Plot training data
fig, ax = plt.subplots(ncols=3, figsize=(20,5))

ax[0].plot(hist.history['total_loss'], color='teal', label='loss')
ax[0].plot(hist.history['val_total_loss'], color='orange', label='val loss')
ax[0].title.set_text('Loss')
ax[0].legend()

ax[1].plot(hist.history['class_loss'], color='teal', label='class loss')
ax[1].plot(hist.history['val_class_loss'], color='orange', label='val class loss')
ax[1].title.set_text('Classification Loss')
ax[1].legend()

ax[2].plot(hist.history['regress_loss'], color='teal', label='regress loss')
ax[2].plot(hist.history['val_regress_loss'], color='orange', label='val regress loss')
ax[2].title.set_text('Regression Loss')
ax[2].legend()

plt.show()
test_data = test.as_numpy_iterator()
# %% make predictions
test_sample = test_data.next()
yhat = facetracker.predict(test_sample[0])

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx in range(4):
    sample_image = test_sample[0][idx]
    sample_coords = yhat[1][idx]

    if yhat[0][idx] > 0.9:
        cv2.rectangle(sample_image,
                        tuple(np.multiply(sample_coords[:2], [120,120]).astype(int)),
                        tuple(np.multiply(sample_coords[2:], [120,120]).astype(int)),
                                (255,0,0),2)

    ax[idx].imshow(sample_image)
# %% Save model

facetracker.save('facetracker.h5')
facetracker = load_model('facetracker.h5')
# %% realtime detection
cap = cv2.VideoCapture(0)
while cap.isOpened():
    _ , frame = cap.read()
    frame = frame[50:500, 50:500,:]
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120,120))
    
    yhat = facetracker.predict(np.expand_dims(resized/255,0))
    sample_coords = yhat[1][0]
    
    if yhat[0] > 0.5: 
        # Controls the main rectangle
        cv2.rectangle(frame, 
                      tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [450,450]).astype(int)), 
                            (255,0,0), 2)
        # Controls the label rectangle
        cv2.rectangle(frame, 
                      tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int), 
                                    [0,-30])),
                      tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                    [80,0])), 
                            (255,0,0), -1)
        
        # Controls the text rendered
        cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                               [0,-5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
    cv2.imshow('EyeTrack', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
# %%
