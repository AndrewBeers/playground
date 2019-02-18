import * as tf from '@tensorflow/tfjs';
import {IMAGE_H, IMAGE_W, MnistData} from './data';


export function createConvModel(layers_deep = 2, cnn_activation = 'relu') {
  // Create a sequential neural network model. tf.sequential provides an API
  // for creating "stacked" models where the output from one layer is used as
  // the input to the next layer.
  const model = tf.sequential();


  for (let i = 0; i < layers_deep; i++) {
    
    if(i == 0){
      model.add(tf.layers.conv2d({
        inputShape: [IMAGE_H, IMAGE_W, 1],
        kernelSize: 3,
        filters: 2,
        activation: cnn_activation
      }));
      model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));
    }else{
      model.add(tf.layers.conv2d({kernelSize: 3, filters: 2, activation: 'relu'}));

      // Max pooling again.
      model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));
    }
    // The first layer of the convolutional neural network plays a dual role:
    // it is both the input layer of the neural network and a layer that performs
    // the first convolution operation on the input. It receives the 28x28 pixels
    // black and white images. This input layer uses 16 filters with a kernel size
    // of 5 pixels each. It uses a simple RELU activation function which pretty
    // much just looks like this: __/

  // 'elu', 'hardSigmoid', 'linear', 'relu', 'relu6', 'selu', 'sigmoid', 'softmax',
// 'softplus', 'softsign', 'tanh'

    // After the first layer we include a MaxPooling layer. This acts as a sort of
    // downsampling using max values in a region instead of averaging.
    // https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks

  // Our third layer is another convolution, this time with 32 filters.

  // Add another conv2d layer.
  model.add(tf.layers.conv2d({kernelSize: 3, filters: 2, activation: cnn_activation}));

  // Now we flatten the output from the 2D filters into a 1D vector to prepare
  // it for input into our last layer. This is common practice when feeding
  // higher dimensional data to a final classification output layer.
  model.add(tf.layers.flatten({}));

  model.add(tf.layers.dense({units: 2, activation: 'relu'}));

  // Our last layer is a dense layer which has 10 output units, one for each
  // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9). Here the classes actually
  // represent numbers, but it's the same idea if you had classes that
  // represented other entities like dogs and cats (two output classes: 0, 1).
  // We use the softmax function as the activation for the output layer as it
  // creates a probability distribution over our 10 classes so their output
  // values sum to 1.
  model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

  return model;
}