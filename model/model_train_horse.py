import tensorflow as tf
import math


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(abs(logs.get('loss'))<0.10):
            print("\nReached 90% accuracy so cancelling training!")
            self.model.stop_training = True

def main():

    import os

    train_horse_dir = os.path.join('../data/horse')
    # train_zebra_dir = os.path.join('./data/train/zebra')

    # valid_horse_dir = os.path.join('/data/train/horse')
    # valid_zebra_dir = os.path.join('/data/train/horse')
    print('The total number of horse images for training : ', len(os.listdir(train_horse_dir)))
    # print('The total number of human images for training : ', len(os.listdir(train_zebra_dir)))
    # print('The total number of horse images for validation : ', len(os.listdir(valid_horse_dir)))
    # print('The total number of human images for validation : ', len(os.listdir(valid_human_dir)))



    # import matplotlib.image as mpimg

    # from matplotlib import pyplot as plt
    # # plotting the images of shape 5x5
    # n_rows = 5
    # n_columns = 5

    # # idx for iterating over the images
    # pic_ind = 0

    # fig = plt.gcf() # get current figure 
    # fig.set_size_inches(n_columns*5, n_rows*5)

    # # get the path for the figures
    # next_horse_pic = [os.path.join(train_horse_dir, fname) for fname in train_horse_names[pic_ind:pic_ind+10]]
    # next_human_pic = [os.path.join(train_zebra_dir, fname) for fname in train_zebra_names[pic_ind:pic_ind+10]]

    # for i, img_path in enumerate(next_horse_pic+next_human_pic):
    #   plt.subplot(n_rows, n_columns, i+1)
    #   plt.axis(False)
    #   img = mpimg.imread(img_path)
    #   plt.imshow(img)

    # plt.show()




    model = tf.keras.Sequential([# First Convolution layer
                                tf.keras.layers.Conv2D(16, (3, 3), activation=tf.nn.relu, input_shape=(300, 300, 3)),
                                tf.keras.layers.MaxPool2D(2, 2),
                                # Second Convolution layer
                                tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
                                tf.keras.layers.MaxPool2D(2, 2),
                                # Third Convolution layer
                                tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
                                tf.keras.layers.MaxPool2D(2, 2),
                                # Fourth Convolution layer
                                tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
                                tf.keras.layers.MaxPool2D(2, 2),
                                # Fifth Convolution layer
                                tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
                                tf.keras.layers.MaxPool2D(2, 2),
                                # Flatten and feed into DNN
                                tf.keras.layers.Flatten(),
                                tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                tf.keras.layers.Dense(1, "sigmoid")])


    from tensorflow.keras.optimizers import RMSprop

    model.compile(optimizer = RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(1/255)
    validation_datagen = ImageDataGenerator(1/255)

    train_generator = train_datagen.flow_from_directory('../data/horse/train',
                                                        target_size=(300, 300),
                                                        batch_size=128,
                                                        class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory('../data/horse/test',
                                                                target_size=(300, 300),
                                                                batch_size=32,
                                                                class_mode='binary')


    callbacks = myCallback()


    history = model.fit(
        train_generator,
        steps_per_epoch=8,
        epochs=15,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=8,
        callbacks=[callbacks]
    )


    import matplotlib.pyplot as plt

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.figure(figsize=(16, 5))
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.figure(figsize=(16, 5))
    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

    print("Saving model")
    model.save('horse_model.h5')


if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # physical_devices = tf.config.experimental.list_physical_devices('CPU')
    print("physical_devices-------------", len(physical_devices))
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    main()