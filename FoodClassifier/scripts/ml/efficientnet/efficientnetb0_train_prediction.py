base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable=False

inputs = layers.Input(shape=INPUT_SHAPE, name="input_layer")

x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D(name="pooling_layer")(x)
x = layers.Dense(len(class_names))(x)

outputs = layers.Activation("softmax", dtype=tf.float32, name="softmax_float32")(x)
model = tf.keras.Model(inputs, outputs)

for layer in model.layers:
    layer.trainable = True
    print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)

for layer in model.layers[1].layers[:20]:
    print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)

model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(0.0001), metrics=['accuracy'])

history_food_classes_fine_tune = model.fit(train_data,
                                                  epochs=100,
                                                  steps_per_epoch=len(train_data),
                                                  validation_data=test_data,
                                                  validation_steps=int(0.15 * len(test_data)),
                                                  callbacks=[create_tensorboard_callback("training_logs", "efficientnetb0_classes_all_data_fine_tuning"),
                                                             model_checkpoint,
                                                             early_stopping,
                                                             reduce_lr])

results_model = model.evaluate(test_data)
print(results_model)

save_dir = "efficientnetb0_model"
model.save(save_dir)

loaded_model = tf.keras.models.load_model(save_dir)

loaded_model.load_weights(checkpoint_path)