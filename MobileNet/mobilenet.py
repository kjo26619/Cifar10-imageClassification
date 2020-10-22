import tensorflow as tf

def bottleneck_block(x, t, c, output_f, s):
    if s == 1:
        shortcut_x = tf.keras.layers.Conv2D(output_f, (1, 1), strides=s, activation=tf.nn.relu6, padding='same')(x)
        conv = tf.keras.layers.Conv2D(t * c, (1, 1), activation=tf.nn.relu6, padding='same')(x)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Conv2D(t * c, (3, 3), strides=s, activation=tf.nn.relu6, padding='same')(conv)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Conv2D(output_f, (1, 1), activation='linear', padding='same')(conv)
        conv = tf.keras.layers.BatchNormalization()(conv)
        x = shortcut_x + conv  # Skip Connection
        out = tf.keras.layers.Activation(tf.nn.relu6)(x)
        out = tf.keras.layers.BatchNormalization()(out)
    else:
        conv = tf.keras.layers.Conv2D(t * c, (1, 1), activation=tf.nn.relu6, padding='same')(x)
        conv = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Conv2D(t * c, (3, 3), strides=s, activation=tf.nn.relu6, padding='same')(conv)
        conv = tf.keras.layers.BatchNormalization()(conv)
        out = tf.keras.layers.Conv2D(output_f, (1, 1), activation='linear', padding='same')(conv)
        out = tf.keras.layers.BatchNormalization()(out)

    return out
    
 def mobile_net_model(shape=(32, 32, 3), feature=32, output_feature=10):
    inputs = tf.keras.Input(shape)

    x = inputs
    conv = tf.keras.layers.Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(x)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(conv)
    conv = tf.keras.layers.BatchNormalization()(conv)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=1, padding='same')(conv)

    x = bottleneck_block(x, 6, 16, 16, 1)
    x = bottleneck_block(x, 6, 24, 24, 1)
    x = bottleneck_block(x, 6, 24, 24, 1)
    
    x = bottleneck_block(x, 6, 32, 32, 1)
    x = bottleneck_block(x, 6, 32, 32, 1)
    x = bottleneck_block(x, 6, 32, 32, 1)
    
    x = bottleneck_block(x, 6, 64, 64, 2)
    x = bottleneck_block(x, 6, 64, 64, 1)
    x = bottleneck_block(x, 6, 64, 64, 1)
    x = bottleneck_block(x, 6, 64, 64, 1)
    
    x = bottleneck_block(x, 6, 96, 96, 1)
    x = bottleneck_block(x, 6, 96, 96, 1)
    x = bottleneck_block(x, 6, 96, 96, 1)
    
    x = bottleneck_block(x, 6, 160, 160, 2)
    x = bottleneck_block(x, 6, 320, 320, 1)
    
    conv = tf.keras.layers.Conv2D(1280, (1, 1), strides=1, padding='same')(x)
    conv = tf.keras.layers.BatchNormalization()(conv)
    x = tf.keras.layers.Activation(tf.nn.relu6)(conv)
  
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    x = tf.keras.layers.Dense(1280, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Dense(1280, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    output = tf.keras.layers.Dense(output_feature, activation='softmax')(x)

    model = tf.keras.Model(inputs, output)
    model.summary()

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
