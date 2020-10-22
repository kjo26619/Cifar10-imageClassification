def vgg_net_model(shape=(32, 32, 3), input_feature=32, output_feature=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(input_feature , (3, 3), activation='relu', input_shape=shape, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(input_feature, (3, 3), activation='relu', input_shape=shape, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Conv2D(input_feature * 2, (3, 3), activation='relu', input_shape=shape, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(input_feature * 2, (3, 3), activation='relu', input_shape=shape, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Conv2D(input_feature * 4, (3, 3), activation='relu', input_shape=shape, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(input_feature * 4, (3, 3), activation='relu', input_shape=shape, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Conv2D(input_feature * 8, (3, 3), activation='relu', input_shape=shape, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(input_feature * 8, (3, 3), activation='relu', input_shape=shape, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(output_feature, activation='softmax')
    ])

    opt = tf.keras.optimizers.Adam(learning_rate=0.00002)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
