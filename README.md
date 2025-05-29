# Description
This is a simple example on how to train a model to recognize 0-9 
images drawn in 28x28 bitmap using Tensor Flow JS.

# Setup Vite App
```
npm create vite@latest digit-draw-app -- --template vanilla-ts
cd digit-draw-app/
npm install @tensorflow/tfjs
```

# How to train the data
## Use ts-train to build the training model
```
cd ts-train
npm install
npm run train
```
## Copy mnist-data/ to ts-visualize-prediction

# How to visualize the data
## Use ts-visualize-data see the mninst data
```
cd ts-visualize-data
npm install
npm run build
npx vite
```

# How to run neural network prediction
## Use ts-visualize-prediction
```
cd ts-visualize-prediction
npm install
npm run dev
```