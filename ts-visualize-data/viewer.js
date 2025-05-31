document.getElementById('fileInput').addEventListener('change', async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const buffer = await file.arrayBuffer();
  const dataView = new DataView(buffer);

  const magic = dataView.getUint32(0);
  const numImages = dataView.getUint32(4);
  const numRows = dataView.getUint32(8);
  const numCols = dataView.getUint32(12);

  const imageData = new Uint8Array(buffer, 16);
  const totalPixels = numRows * numCols;

  const grid = document.getElementById('grid');
  grid.innerHTML = ''; // Clear previous

  const imagesToShow = 100; // You can change this

  for (let i = 0; i < imagesToShow; i++) {
    const canvas = document.createElement('canvas');
    canvas.width = numCols;
    canvas.height = numRows;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(numCols, numRows);

    for (let p = 0; p < totalPixels; p++) {
      const val = imageData[i * totalPixels + p];
      imgData.data[p * 4 + 0] = val;
      imgData.data[p * 4 + 1] = val;
      imgData.data[p * 4 + 2] = val;
      imgData.data[p * 4 + 3] = 255;
    }

    ctx.putImageData(imgData, 0, 0);
    grid.appendChild(canvas);
  }
});
