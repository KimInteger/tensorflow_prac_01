import tensorflow as tf

def create_and_train_model(X_train, y_train, input_channels):
    # 입력 이미지 크기를 명확히 설정
    input_shape = (1920, 1080, input_channels)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),  # 명확한 입력 크기 지정
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=5)
    return model
