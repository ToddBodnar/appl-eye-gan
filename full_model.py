import os

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Conv2DTranspose, Conv2D, UpSampling2D, Reshape, Flatten, MaxPooling2D, LeakyReLU
from keras.optimizers import RMSprop, SGD
from itertools import repeat
import PIL
from datetime import datetime

import numpy as np

training_type = "quickrun"

if training_type == "quickrun":
    prefit_steps = 100
    fit_steps = 100
elif training_type == "normal":
    prefit_steps = 2000
    fit_steps = 25000

partial_batch_size = 16
scale_up = 4

datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=1./255,
        shear_range=0.005,
        zoom_range=0.2,
        horizontal_flip=False,
        fill_mode='nearest')

fake_data_gen = datagen.flow_from_directory(os.getcwd()+'/training_generated/', target_size=(64,64),batch_size=partial_batch_size)

real_data_gen = datagen.flow_from_directory(os.getcwd()+'/training_eye/', target_size=(64,64),batch_size=partial_batch_size)

def output_to_image(array):
    array = array*255
    i = array_to_img(array).resize((64*scale_up,64*scale_up), PIL.Image.ANTIALIAS)
    return i


def save_image(array, name):
    output_to_image(array).save(name)


def get_generator():
    m = Sequential()
    m.add(Conv2D(32,3, input_shape=(64,64,3,), padding='same'))
    m.add(LeakyReLU())
    m.add(MaxPooling2D(2,2))
    m.add(Dropout(0.5))
    m.add(Conv2D(32,3, padding='same'))
    m.add(LeakyReLU())
    m.add(MaxPooling2D(2,2))
    m.add(Dropout(0.5))
    m.add(Conv2D(64,3, padding='same'))
    m.add(LeakyReLU())
    m.add(MaxPooling2D(2,2))
    m.add(Dropout(0.5))
    m.add(Flatten())
    m.add(Dense((1024), input_shape=(100,), activation = 'relu'))
    m.add(Reshape((4,4,int(1024/4/4))))
    m.add(UpSampling2D())
    m.add(Conv2DTranspose(512,8, activation = 'relu', padding='same'))
    m.add(BatchNormalization())
    m.add(UpSampling2D())
    m.add(Conv2DTranspose(256,5, activation = 'relu', padding='same'))
    m.add(BatchNormalization())
    m.add(UpSampling2D())
    m.add(Conv2DTranspose(128,5, activation = 'relu', padding='same'))
    m.add(BatchNormalization())
    m.add(UpSampling2D())
    m.add(Conv2DTranspose(64,5, activation = 'relu', padding='same'))
    m.add(BatchNormalization())
    m.add(Conv2DTranspose(3,5, activation = 'sigmoid', padding='same'))
    return m

get_generator().summary()


def l1_norm(y_true, y_pred):

    from keras import backend as K
    return K.mean(K.abs(y_pred - y_true))

def compile_generator(generator):
    generator.compile(loss=l1_norm, optimizer=RMSprop(lr=0.0001))
    return generator


def fit_generator(generator, iterations = 3500):
    for step in range(0, iterations):
        real = next(fake_data_gen)[0]
        training = [real, real]

        g_err = generator.train_on_batch(real, real)

        if step % 10 == 0:
            sample = generator.predict(real)
            save_image(sample[0], "images/prefitting__%s_%09d.jpg"%(training_type, step))
        print("%s: Step: %d, Generator: %f"%(str(datetime.now()), step, g_err))

def get_descriminator():
    m = Sequential()
    m.add(Conv2D(32,3, input_shape=(64,64,3,), padding='same'))
    m.add(LeakyReLU())
    m.add(MaxPooling2D(2,2))
    m.add(Dropout(0.5))
    m.add(Conv2D(32,3, padding='same'))
    m.add(LeakyReLU())
    m.add(MaxPooling2D(2,2))
    m.add(Dropout(0.5))
    m.add(Conv2D(64,3, padding='same'))
    m.add(LeakyReLU())
    m.add(MaxPooling2D(2,2))
    m.add(Dropout(0.5))
    m.add(Flatten())
    m.add(Dense(1, activation = 'sigmoid'))
    return m

generator = compile_generator(get_generator())

fit_generator(generator, prefit_steps)

def fit_gan(generator, descriminator_builder, sample_image_input, steps = 10000):
    save_image(sample_image_input[0], "images/target_%s.jpg"%(training_type))
    desc = descriminator_builder()
    d = Sequential()
    d.add(desc)
    d.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0005))

    combined_model = Sequential()
    combined_model.add(generator)
    for layer in desc.layers:
            layer.trainable = False
    combined_model.add(desc)
    combined_model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0001))

    descriminator_history = []
    generator_history = []
    pass_through_history = []
    sample_history = []

    for step in range(steps):
        real = next(real_data_gen)[0]
        fake = next(fake_data_gen)[0]

        combined = np.concatenate((real, fake))
        combined_y = np.ones([len(real) + len(fake),1])
        combined_y[len(fake):,:] = 0

        d_err = d.train_on_batch(combined, combined_y)

        y = np.ones([len(fake),1])

        for layer in desc.layers:
            layer.trainable = False
        c_err = combined_model.train_on_batch(fake, y)
        # train twice to make the descriminator's effect stronger
        combined_model.train_on_batch(fake, y)
        for layer in desc.layers:
            layer.trainable = True

        pass_err = generator.train_on_batch(fake, fake)


        sample = generator.predict(fake)

        if step % 50 == 0:
            sample = generator.predict(sample_image_input)

            save_image(sample[0], "images/fitted_%s_%09d.jpg"%(training_type, step))
            generator.save("models/generator_%s_%09d.h5"%(training_type, step))
            d.save("models/descriminator_%s_%09d.h5"%(training_type, step))
            # print(descriminator.predict(x))
            # print(y_all)
            # print(combined.predict(fake_in))
        print("%s: Step: %d, Descriminator: %f, Generator: %f, Pass Through: %f"%(str(datetime.now()), step, d_err, c_err, pass_err))

        descriminator_history.append(d_err)
        generator_history.append(c_err)
        pass_through_history.append(pass_err)
        # sample_history.append(output_to_image(sample[0]))

    return (descriminator_history, generator_history, pass_through_history, sample_history)


descriminator_history, generator_history, pass_through_history, sample_history = fit_gan(generator, get_descriminator, next(fake_data_gen)[0], steps = fit_steps)

fake = next(fake_data_gen)[0]
predicted = generator.predict(fake)

for i in range(0,len(fake)):
    save_image(fake[i], "images/final_fake_%s_%05d.jpg"%(training_type, i))
    save_image(predicted[i], "images/final_adjusted_%s_%05d.jpg"%(training_type, i))
