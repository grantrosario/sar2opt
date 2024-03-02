import tensorflow as tf

import os
import time
import matplotlib.pyplot as plt
import pathlib
from IPython.display import clear_output
from datetime import datetime
from sklearn.metrics import mean_squared_error
from tensorflow_examples.models.pix2pix import pix2pix
from tqdm import tqdm
import numpy as np

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')

#pip install git+https://github.com/tensorflow/examples.git

mse = tf.keras.losses.MeanSquaredError()

AUTOTUNE = tf.data.AUTOTUNE

BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

LAMBDA = 10

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5, epsilon=1e-07)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5, epsilon=1e-07)
discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5, epsilon=1e-07)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5, epsilon=1e-07)

generator_g_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(generator_g_optimizer, dynamic=True)
generator_f_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(generator_f_optimizer, dynamic=True)
discriminator_x_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(discriminator_x_optimizer, dynamic=True)
discriminator_y_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(discriminator_y_optimizer, dynamic=True)

log_dir="sar2opt/s_cyclegan/logs/"

summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

def load_img(image_file):
  # Read and decode an image file to a uint8 tensor
  image = tf.io.read_file(image_file)
  image = tf.io.decode_jpeg(image)

  # Split each image tensor into two tensors:
  # - one with a real building facade image
  # - one with an architecture label image 
  w = tf.shape(image)[1]
  w = w // 2
  sar_image = image[:, w:, :]
  opt_image = image[:, :w, :]

  # Convert both images to float32 tensors
  sar_image = tf.cast(sar_image, tf.float16)
  opt_image = tf.cast(opt_image, tf.float16)
    
  return sar_image, opt_image


def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image


def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]


# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image):
  # Resizing to 286x286
  input_image, real_image = resize(input_image, real_image, 286, 286)

  # Random cropping back to 256x256
  input_image, real_image = random_crop(input_image, real_image)

  if tf.random.uniform(()) > 0.5:
    # Random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image


def preprocess_image_train(image_file):
  input_image, real_image = load_img(image_file)
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image


def preprocess_image_test(image_file):
  input_image, real_image = load_img(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image


def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)

  generated_loss = loss_obj(tf.zeros_like(generated), generated)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss * 0.5


def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

  return LAMBDA * loss1


def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss


def generate_images(model, test_input, tar, idx):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  test_input = test_input.numpy().astype('float32')
  tar = tar.numpy().astype('float32')
  prediction = prediction.numpy().astype('float32')

  print('\n')
  print(np.isnan(prediction).any())
  print('\n')

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.savefig(f'data/compare_{idx}.png', bbox_inches='tight')
  plt.close()


@tf.function
def train_step(real_x, real_y, step):
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.
  with tf.GradientTape(persistent=True) as tape:
    # Generator G translates X -> Y
    # Generator F translates Y -> X.

    fake_y = generator_g(real_x, training=True)
    cycled_x = generator_f(fake_y, training=True)

    fake_x = generator_f(real_y, training=True)
    cycled_y = generator_g(fake_x, training=True)

    # same_x and same_y are used for identity loss.
    same_x = generator_f(real_x, training=True)
    same_y = generator_g(real_y, training=True)

    disc_real_x = discriminator_x(real_x, training=True)
    disc_real_y = discriminator_y(real_y, training=True)

    disc_fake_x = discriminator_x(fake_x, training=True)
    disc_fake_y = discriminator_y(fake_y, training=True)

    # calculate the loss
    gen_g_loss = generator_loss(disc_fake_y)
    gen_f_loss = generator_loss(disc_fake_x)

    total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

    # Total generator loss = adversarial loss + cycle loss
    total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y) + mse(real_y, fake_y)
    total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x) + mse(real_x, fake_x)

    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Loss scaling for mixed precision
    total_gen_g_loss_scaled = generator_g_optimizer.get_scaled_loss(total_gen_g_loss)
    total_gen_f_loss_scaled = generator_f_optimizer.get_scaled_loss(total_gen_f_loss)
    disc_x_loss_scaled = discriminator_x_optimizer.get_scaled_loss(disc_x_loss)
    disc_y_loss_scaled = discriminator_y_optimizer.get_scaled_loss(disc_y_loss)

  # Calculate the gradients for generator and discriminator
  generator_g_gradients_scaled = tape.gradient(total_gen_g_loss_scaled, 
                                        generator_g.trainable_variables)
  generator_f_gradients_scaled = tape.gradient(total_gen_f_loss_scaled, 
                                        generator_f.trainable_variables)

  discriminator_x_gradients_scaled = tape.gradient(disc_x_loss_scaled, 
                                            discriminator_x.trainable_variables)
  discriminator_y_gradients_scaled = tape.gradient(disc_y_loss_scaled, 
                                            discriminator_y.trainable_variables)

  generator_g_gradients = generator_g_optimizer.get_unscaled_gradients(generator_g_gradients_scaled)
  generator_f_gradients = generator_f_optimizer.get_unscaled_gradients(generator_f_gradients_scaled)
  discriminator_x_gradients = discriminator_x_optimizer.get_unscaled_gradients(discriminator_x_gradients_scaled)
  discriminator_y_gradients = discriminator_y_optimizer.get_unscaled_gradients(discriminator_y_gradients_scaled)

  # Apply the gradients to the optimizer
  generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                            generator_g.trainable_variables))

  generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                            generator_f.trainable_variables))

  discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))

  discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_g_total_loss', total_gen_g_loss, step)
    tf.summary.scalar('gen_g_loss', gen_g_loss, step)
    tf.summary.scalar('total_cycle_loss', total_cycle_loss, step)
    tf.summary.scalar('disc_x_loss', disc_x_loss, step)
    tf.summary.scalar('disc_y_loss', disc_y_loss, step)


def main():

  train_path = pathlib.Path('data/train')
  test_path = pathlib.Path('data/test')
  train_filenames = tf.constant([os.path.join(train_path, fname) for fname in os.listdir(train_path)])
  test_filenames = tf.constant([os.path.join(test_path, fname) for fname in os.listdir(test_path)])
  train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames))
  test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames))

  train_dataset = train_dataset.cache().map(
  	preprocess_image_train, num_parallel_calls=AUTOTUNE).batch(
  	BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

  test_dataset = test_dataset.cache().map(
  	preprocess_image_test, num_parallel_calls=AUTOTUNE).batch(
  	BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

  sample_sar, sample_opt = next(iter(train_dataset))
  # plt.imshow(sample_sar[0])
  # plt.show()

  checkpoint_path = "sar2opt/s_cyclegan/training_checkpoints"
  checkpoint_prefix = os.path.join(checkpoint_path, "ckpt")

  ckpt = tf.train.Checkpoint(generator_g=generator_g,
                             generator_f=generator_f,
                             discriminator_x=discriminator_x,
                             discriminator_y=discriminator_y,
                             generator_g_optimizer=generator_g_optimizer,
                             generator_f_optimizer=generator_f_optimizer,
                             discriminator_x_optimizer=discriminator_x_optimizer,
                             discriminator_y_optimizer=discriminator_y_optimizer)

  ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

  # if a checkpoint exists, restore the latest checkpoint.
  if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')

  generate_images(generator_g, sample_sar, sample_opt, 0)

  EPOCHS = 25

  tf.config.run_functions_eagerly(True)

  for epoch in range(EPOCHS):
    start = time.time()

    n = 0
    for image_x, image_y in tqdm(tf.data.Dataset.zip((train_dataset))):
      train_step(image_x, image_y, epoch)
      # if n % 10 == 0:
      #   print ('.', end='')
      n += 1

    clear_output(wait=True)
    # Using a consistent image (sample_horse) so that the progress of the model
    # is clearly visible.
    generate_images(generator_g, sample_sar, sample_opt, epoch+1)

    #if (epoch + 1) % 5 == 0: 
    ckpt.save(file_prefix=checkpoint_prefix)
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                           checkpoint_prefix))

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))

if __name__ == "__main__":
	main()




