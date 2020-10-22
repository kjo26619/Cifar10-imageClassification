import tensorflow as tf

def residual_block(x, input_feature, output_feature, stride):
    if stride == 2:
        shortcut_x = tf.keras.layers.Conv2D(output_feature, (1, 1), strides=stride, activation='relu', padding='same')(
            x)
        shortcut_x = tf.keras.layers.BatchNormalization()(shortcut_x)
        conv = tf.keras.layers.Conv2D(input_feature, (1, 1), strides=stride, activation='relu', padding='same')(x)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Conv2D(input_feature, (3, 3), activation='relu', padding='same')(conv)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Conv2D(output_feature, (1, 1), activation='relu', padding='same')(conv)
        conv = tf.keras.layers.BatchNormalization()(conv)

        x = shortcut_x + conv
        x = tf.keras.layers.Activation('relu')(x)
    else:
        shortcut_x = x
        conv = tf.keras.layers.Conv2D(input_feature, (1, 1), activation='relu', padding='same')(x)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Conv2D(input_feature, (3, 3), activation='relu', padding='same')(conv)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Conv2D(output_feature, (1, 1), activation='relu', padding='same')(conv)
        conv = tf.keras.layers.BatchNormalization()(conv)
        x = shortcut_x + conv  # Skip Connection
        x = tf.keras.layers.Activation('relu')(x)

    return x
   
def res_net50_model(shape=(32, 32, 3), input_feature=32, output_feature=10):
    inputs = tf.keras.Input(shape)
    x = inputs

    conv = tf.keras.layers.Conv2D(input_feature, (3, 3), strides=1, activation='relu', padding='same')(x)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.MaxPooling2D((3, 3), strides=1)(conv)
    conv = tf.keras.layers.Activation('relu')(conv)

    conv = tf.keras.layers.Conv2D(input_feature * 2, (3, 3), strides=1, activation='relu', padding='same')(conv)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.MaxPooling2D((3, 3), strides=1)(conv)
    conv = tf.keras.layers.Activation('relu')(conv)

    x = tf.keras.layers.Conv2D(256, (3, 3), strides=1, activation='relu', padding='same')(conv)
    x = residual_block(x, input_feature=64, output_feature=256, stride=1)
    x = residual_block(x, input_feature=64, output_feature=256, stride=1)
    x = residual_block(x, input_feature=64, output_feature=256, stride=1)

    x = residual_block(x, input_feature=128, output_feature=512, stride=2)
    x = residual_block(x, input_feature=128, output_feature=512, stride=1)
    x = residual_block(x, input_feature=128, output_feature=512, stride=1)

    x = residual_block(x, input_feature=256, output_feature=1024, stride=2)
    x = residual_block(x, input_feature=256, output_feature=1024, stride=1)
    x = residual_block(x, input_feature=256, output_feature=1024, stride=1)

    x = tf.keras.layers.GlobalAvgPool2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    output = tf.keras.layers.Dense(output_feature, activation='softmax')(x)

    model = tf.keras.Model(inputs, output)
    model.summary()

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
