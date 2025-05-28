const IMAGE_SIZE = 28;
const NUM_IMAGES = 100;
const IMAGE_BYTES = IMAGE_SIZE * IMAGE_SIZE;

async function loadMnistData() {
  const [imgRes, labelRes] = await Promise.all([
    fetch("/train-images-idx3-ubyte"),
    fetch("/train-labels-idx1-ubyte"),
  ]);

  const imgBuf = await imgRes.arrayBuffer();
  const labelBuf = await labelRes.arrayBuffer();

  return {
    images: new Uint8Array(imgBuf, 16), // Skip the 16-byte header
    labels: new Uint8Array(labelBuf, 8), // Skip the 8-byte header
  };
}

async function drawMnist() {
  const { images, labels } = await loadMnistData();

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d")!;
  const scale = 2;
  const cols = 10;
  const rows = 10;
  canvas.width = cols * IMAGE_SIZE * scale;
  canvas.height = rows * IMAGE_SIZE * scale;
  document.body.appendChild(canvas);

  for (let i = 0; i < NUM_IMAGES; i++) {
    const x = (i % cols) * IMAGE_SIZE * scale;
    const y = Math.floor(i / cols) * IMAGE_SIZE * scale;

    const imageData = ctx.createImageData(IMAGE_SIZE, IMAGE_SIZE);
    const offset = i * IMAGE_BYTES;

    for (let j = 0; j < IMAGE_BYTES; j++) {
      const val = images[offset + j];
      imageData.data[j * 4 + 0] = val; // R
      imageData.data[j * 4 + 1] = val; // G
      imageData.data[j * 4 + 2] = val; // B
      imageData.data[j * 4 + 3] = 255; // A
    }

    const tmpCanvas = document.createElement("canvas");
    tmpCanvas.width = IMAGE_SIZE;
    tmpCanvas.height = IMAGE_SIZE;
    tmpCanvas.getContext("2d")!.putImageData(imageData, 0, 0);

    ctx.drawImage(tmpCanvas, x, y, IMAGE_SIZE * scale, IMAGE_SIZE * scale);

    ctx.fillStyle = "black";
    ctx.font = "12px sans-serif";
    ctx.fillText(labels[i].toString(), x + 2, y + 12);
  }
}

drawMnist();
