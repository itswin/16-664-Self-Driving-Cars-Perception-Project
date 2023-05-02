from glob import glob
import numpy as np
import tensorflow as tf
import os
import argparse
import math
import csv
from tensorflow.keras import datasets, layers, models
import matplotlib.image as mpimg
import skimage

classes = (
    'Unknown', 'Compacts', 'Sedans', 'SUVs', 'Coupes',
    'Muscle', 'SportsClassics', 'Sports', 'Super', 'Motorcycles',
    'OffRoad', 'Industrial', 'Utility', 'Vans', 'Cycles',
    'Boats', 'Helicopters', 'Planes', 'Service', 'Emergency',
    'Military', 'Commercial', 'Trains'
)

w = 300
h = 600
num_classes = 23

def class_to_label(class_id):
    if 1 <= class_id <= 8:
        return 1
    elif 9 <= class_id <= 15:
        return 2
    else:
        return 0

def write_test_labels(classes, num_images=10000, start=0, no_header=False):
    images = glob('test/*/*_image.jpg')
    images.sort()

    name = 'test_labels2.csv'
    with open(name, 'w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        if not no_header:
            writer.writerow(['guid/image', 'label'])

        for (file, class_id) in zip(images[start:start + num_images], classes):
            guid = file.split('/')[-2]
            idx = file.split('/')[-1].replace('_image.jpg', '')
            max_class_id = np.argmax(class_id)
            label = class_to_label(max_class_id)
            # print(label, max_class_id, class_id)
            
            writer.writerow(['{}/{}'.format(guid, idx), label])

    print('Wrote report file `{}`'.format(name))

def load_images(path, num_images=10000, start=0):
    global w, h, num_classes

    images = glob('{}/*/*_image.jpg'.format(path))
    images.sort()
    imgs = []

    for image_file in images[start:start+num_images]:
        img = np.asarray(mpimg.imread(image_file))
        img = skimage.transform.resize(img, (w, h))
        img = img / 255.0
        imgs.append(img)

        if len(imgs) % 100 == 0:
            print(f"Read {len(imgs)} images...")

    return np.asarray(imgs)

def load_data(path, num_images=10000, start=0):
    global w, h, num_classes

    images = glob('{}/*/*_image.jpg'.format(path))
    images.sort()
    bboxes = glob('{}/*/*_bbox.bin'.format(path))
    bboxes.sort()

    imgs = []
    classes = []

    for (image_file, bbox_file) in zip(images[start:start + num_images], bboxes[start:start + num_images]):
        bbox = np.fromfile(bbox_file, dtype=np.float32)
        bbox = bbox.reshape([-1, 11])
        found_valid = False
        for b in bbox:
            # ignore_in_eval
            if bool(b[-1]):
                break
            found_valid = True
            class_id = b[9].astype(np.uint8)
        if not found_valid:
            class_id = 0

        img = np.asarray(mpimg.imread(image_file))
        img = skimage.transform.resize(img, (w, h))
        img = img / 255.0
        imgs.append(img)
        class_id = class_to_label(class_id)
        classes.append(class_id)

        if len(imgs) % 100 == 0:
            print(f"Read {len(imgs)} images...")

    return np.asarray(imgs), np.asarray(classes)

def create_model():
    global w, h, num_classes

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(w, h, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes))

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='Trainer',
                    description='Trainer for 16-664 Project')

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--checkpoint', action='store_true')

    args = parser.parse_args()

    model = create_model()
    model.summary()

    checkpoint_path = "training_1/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    if args.train:
        train_images, train_labels = load_data("trainval")
        print(f"Loaded {len(train_images)} images")

        batch_size = 32

        # Calculate the number of batches per epoch
        n_batches = len(train_images) / batch_size
        n_batches = math.ceil(n_batches)
        save_freq_epochs = 10

        # Create a callback that saves the model's weights every 5 epochs
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, 
            verbose=1, 
            save_weights_only=True,
            save_freq=save_freq_epochs*n_batches)

        if args.checkpoint:
            latest = tf.train.latest_checkpoint(checkpoint_dir)
            print(f"Loading latest checkpoint: {latest}")
            model.load_weights(latest)
        else:
            print(f"Saving initial checkpoint: {checkpoint_path.format(epoch=0)}")
            model.save_weights(checkpoint_path.format(epoch=0))

        history = model.fit(train_images, 
                            train_labels, 
                            epochs=args.epochs, 
                            callbacks=[cp_callback])
    elif args.test:
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        print(f"Loading latest checkpoint: {latest}")
        model.load_weights(latest)

        test_images = load_images("test")
        print(f"Loaded {len(test_images)} images")

        classes = model.predict(test_images)
        write_test_labels(classes)
