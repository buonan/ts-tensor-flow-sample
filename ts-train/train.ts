import * as tf from '@tensorflow/tfjs-node';
import * as mnist from 'mnist-data';

async function loadMnistData() {
  //const data = mnist.training(0, 10000); // First 10,000 samples
  const data = mnist.training(0, 60000); // Full dataset (better for training)

  const rawImages = data.images.values; // 28x28 arrays
  const rawLabels = data.labels.values; // integers 0–9

  // Flatten 2D 28x28 pixels images to 1D vectors of 784 length and
  // Normalize pixel values from 0–255 to 0–1
  const images: number[][] = rawImages.map((img2d: any) =>
    img2d.flat().map((p: any) => p / 255) 
  );

  // Labels are integers 0–9, convert to one-hot encoded vectors
  // e.g. 3 becomes [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
  // Use tf.oneHot to convert labels to one-hot encoded tensors
  // The labels are already in the range 0–9, so we can use them directly
  // tf.oneHot expects a tensor of integers and the number of classes (10 for MNIST)
  // Convert rawLabels to a tensor of integers
  // and then apply one-hot encoding
  // to create a tensor of shape [numSamples, 10]
  // where numSamples is the number of images (10000 in this case)
  // Each row corresponds to one image and contains a one-hot encoded vector
  const labels = tf.tensor1d(rawLabels, 'int32');
  const oneHotLabels = tf.oneHot(labels, 10);

  const xs = tf.tensor2d(images, [images.length, 784]);

  return { xs, ys: oneHotLabels, rawLabels };
}

function showAsciiImage(flatImage: number[], width = 28) {
  const chars = ' .:-=+*#%@'; // 0 → space, 1 → @
  for (let i = 0; i < flatImage.length; i += width) {
    const row = flatImage.slice(i, i + width)
      .map(v => chars[Math.floor(v * (chars.length - 1))])
      .join('');
    console.log(row);
  }
}

// Define a better model architecture
function cnnModel() {
  const model = tf.sequential();

  // Reshape 784 -> 28x28x1
  model.add(tf.layers.reshape({ 
    targetShape: [28, 28, 1], 
    inputShape: [784] 
  }));

  model.add(tf.layers.conv2d({
    filters: 32,
    kernelSize: 3,
    activation: 'relu',
    padding: 'same'
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

  model.add(tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    activation: 'relu',
    padding: 'same'
  }));
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

  model.add(tf.layers.flatten());

  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

// Working model
function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [784], units: 256, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}

async function trainAndEvaluate() {
  const { xs, ys, rawLabels } = await loadMnistData();

  const sampleIndex = 0;

  console.log(`Label: ${rawLabels[sampleIndex]}`);
  const imageData = xs.slice([sampleIndex, 0], [1, 784]).dataSync();
  showAsciiImage(Array.from(imageData));

  // Label distribution sanity check
  const labelDist = rawLabels.reduce((acc: any, l: any) => {
    acc[l] = (acc[l] || 0) + 1;
    return acc;
  }, {} as Record<number, number>);
  console.log('Label distribution:', labelDist);

  const model = createModel();  // working model

  await model.fit(xs, ys, {
    epochs: 20,
    batchSize: 64,
    validationSplit: 0.1,
    shuffle: true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(`Epoch ${epoch + 1}: loss=${logs?.loss?.toFixed(4)}, accuracy=${(logs?.acc ?? logs?.accuracy)?.toFixed(4)}`);
      },
    },
  });

  await model.save('file://./mnist-model');

  // Optional: test prediction
  const predictionTensor = model.predict(xs.slice([0, 0], [10, 784])) as tf.Tensor;
  const predictions = predictionTensor.argMax(-1).dataSync();
  console.log('Predictions for first 10 samples:', predictions);
  console.log('Actual labels:', rawLabels.slice(0, 10));
}

trainAndEvaluate().catch(err => console.error(err));
