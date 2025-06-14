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
const uploadInput = document.getElementById('upload') as HTMLInputElement;
uploadInput.addEventListener('change', handleImageUpload);

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
  const outputDiv = document.getElementById('softmax-output')!;
  let probs = new Float32Array(10).fill(0);
  outputDiv.innerHTML = Array.from(probs).map((p, i) =>
    `${i}: ${(p * 100).toFixed(2)}%`
  ).join('<br>');
});

// Load your saved model hosted somewhere or local file
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
  predictionText.textContent = 'Model loaded! Draw a digit.';
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

function handleImageUpload(event: Event) {
  const input = event.target as HTMLInputElement;
  if (!input.files || input.files.length === 0) return;

  const file = input.files[0];
  const reader = new FileReader();
  const img = new Image();

  reader.onload = function (e) {
    img.onload = function () {
      // Clear the canvas
      ctx.fillStyle = 'white';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Draw the uploaded image resized into the canvas
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      predict(); // Run prediction after image is drawn
    };

    img.src = e.target?.result as string;
  };

  reader.readAsDataURL(file);
}

async function predict() {
  if (!model) return;
  predictionText.textContent = 'Predicting...';
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
