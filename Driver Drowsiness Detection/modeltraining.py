import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Input, Flatten, Dense, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Data Augumentation

tf.config.list_physical_devices('GPU')

batchsize = 8

train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=0.2, shear_range=0.2, zoom_range=0.2,
                                   width_shift_range=0.2, height_shift_range=0.2, validation_split=0.2)

train_data = train_datagen.flow_from_directory(r'C:\Users\VENKAT\Downloads\archive\data\train', target_size=(80, 80),
                                               batch_size=batchsize, class_mode='categorical', subset='training')

validation_data = train_datagen.flow_from_directory(r'C:\Users\VENKAT\Downloads\archive\data\train',
                                                    target_size=(80, 80), batch_size=batchsize,
                                                    class_mode='categorical', subset='validation')

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_data = test_datagen.flow_from_directory(r'C:\Users\VENKAT\Downloads\archive\data\test', target_size=(80, 80),
                                             batch_size=batchsize, class_mode='categorical')

bmodel = InceptionV3(include_top=False, weights='imagenet', input_tensor=Input(shape=(80, 80, 3)))
hmodel = bmodel.output
hmodel = Flatten()(hmodel)
hmodel = Dense(64, activation='relu')(hmodel)
hmodel = Dropout(0.5)(hmodel)
hmodel = Dense(2, activation='softmax')(hmodel)

model = Model(inputs=bmodel.input, outputs=hmodel)
for layer in bmodel.layers:
    layer.trainable = False

model.summary()

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint(r'C:\Users\VENKAT\Desktop\New folder\model.h5', monitor='val_loss',
                             save_best_only=True, verbose=3)

earlystop = EarlyStopping(monitor='val_loss', patience=7, verbose=3, restore_best_weights=True)

learning_rate = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=3, )

callbacks = [checkpoint, earlystop, learning_rate]

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_data, steps_per_epoch=train_data.samples // batchsize, validation_data=validation_data,
          validation_steps=validation_data.samples // batchsize, callbacks=callbacks, epochs=5)

# Model Evaluation

acc_tr, loss_tr = model.evaluate(train_data)
print(acc_tr)
print(loss_tr)

acc_vr, loss_vr = model.evaluate(validation_data)
print(acc_vr)
print(loss_vr)

acc_test, loss_test = model.evaluate(test_data)
print(acc_tr)
print(loss_tr)
