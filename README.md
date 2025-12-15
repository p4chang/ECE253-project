To apply the image processing techniques to the images, we simply run runPipeline.m by uncommenting/commenting out the corresponding image processing technique as needed. Then we also modify the enhancementCategory to the output file names as desired. 

Here are the [Testing/validation data](https://drive.google.com/drive/folders/1WNtBGoE5hpt5G9ctnq9GOtIGLZfVIWKQ?usp=sharing), which includes a zipped folder of all images after image processing was applied. Our final test dataset consists of 46 images we took and 300 images selected from this online traffic sign dataset [Roboflow](https://universe.roboflow.com/us-traffic-signs-pwkzx/us-road-signs).

The two models we use are pretrained versions of ResNet50 and Vision Transformer. These models are trained on the ImageNet 1000 dataset and are available on pytorch. The Jupyter notebook to run the model pipeline can be found in model.ipynb

