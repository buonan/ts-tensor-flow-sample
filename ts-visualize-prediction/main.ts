import * as tf from '@tensorflow/tfjs';

const canvas = document.getElementById('draw-canvas') as HTMLCanvasElement;
const ctx = canvas.getContext('2d')!;
const predictionText = document.getElementById('prediction')!;
const clearBtn = document.getElementById('clear-btn')!;

let isDrawing = false;

// Set up canvas for drawing
ctx.lineWidth = 20;
ctx.lineCap = 'round';
ctx.strokeStyle = 'black';
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = 'black';


canvas.addEventListener('pointerdown', () => { isDrawing = true; });
canvas.addEventListener('pointerup', () => { isDrawing = false; ctx.beginPath(); predict(); });
canvas.addEventListener('pointerout', () => { isDrawing = false; ctx.beginPath(); });
canvas.addEventListener('pointermove', draw);

function draw(event: PointerEvent) {
  if (!isDrawing) return;
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, y);
}

clearBtn.addEventListener('click', () => {
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  predictionText.textContent = 'Prediction: -';
});

// Load your saved model hosted somewhere or local file
// For demo: Use pre-trained mnist model from tfjs repository
const MODEL_URL = './mnist-model/model.json'; // Change this to your model path
//const MODEL_URL = 'https://tfhub.dev/tensorflow/tfjs-model/mnist/digit_classifier/1/default/1';

let model: tf.LayersModel;

async function loadModel() {
  predictionText.textContent = 'Loading model...';
  // Uncomment the next line to load from a local file  
  model = await tf.loadLayersModel('./mnist-model/model.json');
  // Uncomment the next line to load from TensorFlow Hub
  // Note: This requires a server setup to serve the model files correctly
  // Uncomment the next line to load from TensorFlow Hub
  // model = await tf.loadLayersModel('https://tfhub.dev/tensorflow/tfjs-model/mnist/digit_classifier/1/default/1');
  // Uncomment the next line to load from a local file
  // model = await tf.loadLayersModel('file://./mnist-model/model.json');
  // Uncomment the next line to load from TensorFlow Hub  
  //model = await tf.loadLayersModel(MODEL_URL, { fromTFHub: true });
  predictionText.textContent = 'Model loaded! Draw a digit.';
}

function preprocessCanvas1(image: HTMLCanvasElement) {
  // Resize to 28x28, grayscale, normalize [0,1]
  return tf.tidy(() => {
    const tensor = tf.browser.fromPixels(image, 1); // grayscale
    const resized = tf.image.resizeBilinear(tensor, [28, 28]);
    const normalized = resized.div(255.0);
    return normalized.reshape([1, 28, 28, 1]);
  });
}

async function preprocessCanvasSmart(canvas: HTMLCanvasElement): Promise<tf.Tensor> {
  const image = tf.browser.fromPixels(canvas, 1).toFloat().div(255.0); // [280,280,1]
  const inverted = tf.scalar(1).sub(image); // invert to match MNIST white-on-black
  const squeezed = inverted.squeeze(); // [280,280]

  const condition = squeezed.greater(0.15); // threshold
  const nonZeroCoords = await tf.whereAsync(condition); // [N, 2]

  if (nonZeroCoords.shape[0] === 0) {
    // No digit drawn
    return tf.zeros([1, 784]);
  }

  // Extract min/max box
  const [ys, xs] = tf.split(nonZeroCoords, 2, 1);
  const yMin = ys.min().arraySync() as number;
  const yMax = ys.max().arraySync() as number;
  const xMin = xs.min().arraySync() as number;
  const xMax = xs.max().arraySync() as number;

  const boxHeight = yMax - yMin + 1;
  const boxWidth = xMax - xMin + 1;
  const size = Math.max(boxHeight, boxWidth);
  const centerY = (yMin + yMax) / 2;
  const centerX = (xMin + xMax) / 2;
  const halfSize = size / 2;

  const top = Math.max(0, Math.floor(centerY - halfSize));
  const left = Math.max(0, Math.floor(centerX - halfSize));
  const bottom = Math.min(280, top + size);
  const right = Math.min(280, left + size);

  const box = [top / 280, left / 280, bottom / 280, right / 280];

  // Crop and resize inside tidy
  const cropped = tf.tidy(() => {
    const batched = inverted.expandDims(0); // shape [1, 280, 280, 1]
    const crop = tf.image.cropAndResize(batched as tf.Tensor4D, [box], [0], [28, 28]); // returns [1, 28, 28, 1]
    return crop.reshape([1, 784]); // final shape [1, 784]
  });


  return cropped;
}

async function predict() {
  if (!model) return;

  const input = await preprocessCanvasSmart(canvas);
  const prediction = model.predict(input) as tf.Tensor;

  const probs = await prediction.data();

  const outputDiv = document.getElementById('softmax-output')!;
  outputDiv.innerHTML = Array.from(probs).map((p, i) =>
    `${i}: ${(p * 100).toFixed(2)}%`
  ).join('<br>');

  prediction.print();

  const digit = (await prediction.argMax(-1).data())[0];
  predictionText.textContent = `Prediction: ${digit}`;

  input.dispose();
  prediction.dispose();
}


loadModel();
