import * as tf from '@tensorflow/tfjs-node';
import * as mnist from 'mnist-data';

/**
 * Loads and processes MNIST data.
 */
function loadData() {
  const train = mnist.training(0, 60000);
  const test = mnist.testing(0, 10000);

  const process = (images: number[][][], labels: number[]) => {
    // Normalize pixel values (0–1) and keep shape [n, 28, 28, 1]
    const normalizedImages = images.map(img =>
      img.map(row =>
        row.map(p => [p / 255]) // <- convert number to [number] to add channel dim
      )
    );


    const xs = tf.tensor4d(normalizedImages, [images.length, 28, 28, 1]);
    const labelsTensor = tf.tensor1d(labels, 'int32');
    const ys = tf.oneHot(labelsTensor, 10);

    return { xs, ys };
  };

  const trainData = process(train.images.values as number[][][], train.labels.values as number[]);
  const testData = process(test.images.values as number[][][], test.labels.values as number[]);

  return {
    ...trainData,
    testXs: testData.xs,
    testYs: testData.ys,
    rawTestLabels: test.labels.values as number[]
  };
}

/**
 * Creates a CNN model for MNIST classification.
 */
function createCnnModel(): tf.LayersModel {
  const model = tf.sequential();

  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    filters: 32,
    kernelSize: 3,
    activation: 'relu',
    padding: 'same'
  }));

  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

  model.add(tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    activation: 'relu',
    padding: 'same'
  }));

  model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

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

/**
 * Trains and evaluates the CNN model.
 */
async function trainAndEvaluate(): Promise<void> {
  const { xs, ys, testXs, testYs, rawTestLabels } = loadData();
  const model = createCnnModel();

  model.summary();

  await model.fit(xs, ys, {
    epochs: 20,
    batchSize: 64,
    validationSplit: 0.1,
    shuffle: true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        const loss = logs?.loss?.toFixed(4);
        const acc = (logs?.acc ?? logs?.accuracy)?.toFixed(4);
        console.log(`Epoch ${epoch + 1}: loss=${loss}, accuracy=${acc}`);
      },
    }
  });

  const evalOutput = await model.evaluate(testXs, testYs) as tf.Scalar | tf.Scalar[];
  const accTensor = Array.isArray(evalOutput) ? evalOutput[1] : evalOutput;
  const accValue = (await accTensor.data())[0];
  console.log(`\nTest Accuracy: ${accValue.toFixed(4)}`);

  const predictions = model.predict(testXs.slice([0, 0, 0, 0], [10, 28, 28, 1])) as tf.Tensor;
  const predictedLabels = predictions.argMax(-1).dataSync();
  console.log('Predicted:', predictedLabels);
  console.log('Actual:   ', rawTestLabels.slice(0, 10));

  await model.save('file://./mnist-cnn-model');
}

trainAndEvaluate().catch(err => console.error(err));
