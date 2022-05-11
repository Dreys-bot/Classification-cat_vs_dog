# Dogs vs Cats
In this repository, we will try to build an image classification prediction model based on the Convolutional Neural Networks architecture using the MXNet library for Deep Learning.
The dataset consists of 3000 images of dogs and cats. The images provided have different sizes.


## Preparing the datasets
Our goal is to generate two files, train_file for training and validation_file for validation, and the first one contains 95% images.
This can be done with the following steps.
![Architecture](https://github.com/Dreys-bot/Classification-cat_vs_dog/blob/main/steps.png)

First download the train dataset from [ here ](https://drive.google.com/open?id=0B0ChXT-rp95aa1FHa3ZmVEJLZkk) and unpack it.

Then create the corresponding directories and make sure that the images belonging to the same class are placed in the same directory:

* cats - in the directory named `data/full_train_data/cat`
* dogs - in the directory named `data/full_train_data/dog

Make sure to create output directories where the generated datasets will be stored.


After that we will prepare two `.lst` files, which consist of labels and image paths that can be used to generate image files.


As a result, two files with image lists will be generated:

* `data/data_set_train/train_file.lst` - train data list
* `data/data_set_train/validation_file.lst` - validation data list

Class labels will be generated: `0 - cat, 1 - dog`

Finally we will generate the `rec files` . We will resize the images so that the short edge is at least 32 pixels and save them with a quality of 95/100. We will also use 16 threads to speed up the packaging. I believe that larger image sizes can improve prediction accuracy, but this will result in a significant increase in learning time, which is prohibitive in my hardware setup.


The resulting records will be generated in the directory: `data/data_set/`

Before moving on to training, display the images of the dogs and cats:

![image](https://github.com/Dreys-bot/Classification-cat_vs_dog/blob/main/chiens.png)

![image](https://github.com/Dreys-bot/Classification-cat_vs_dog/blob/main/chats.png)

## Run the training
Start the training with the following command

I believe that more layers can improve the accuracy of predictions, but this will result in a dramatic increase in training time, which is prohibitive in my hardware setup.

The example output for the first few runs of the four epochs is as follows:

![epochs](https://github.com/Dreys-bot/Classification-cat_vs_dog/blob/main/epoques.png)

As can be seen, the validation accuracy increases, but the prediction model must converge to ensure that our CNN is learning something. Unfortunately, due to hardware limitations, I could not test all 300 epochs to see if the model converges, but interested parties can try this with a better hardware configuration. With my hardware configuration (CPU only, Intel Core i3 at 2.5 GHz, 8 GB RAM, SSD), training on 300 epochs would take about 740 hours (or a month). I think in the CUDA-capable configuration, it will take hours instead.


## Run the prediction on the trained model
Once the model training is complete, we will prepare a test data set. And then make predictions on it.

First download the test data files from [ here ](https://drive.google.com/open?id=0B0ChXT-rp95aUGRnZTgyUWpjNDg) and unzip them.

Then create a list of images in the dataset
Load the prediction model, then run the prediction on the test data set and the trained model

```python
from tensorflow.keras.preprocessing import image
img = image.load_img('/tmp/dog2.jpg',target_size = (150,150))
x = image.img_to_array(img)
x = x[np.newaxis]
```

![result](https://github.com/Dreys-bot/Classification-cat_vs_dog/blob/main/results.png)



## Results

The result of the prediction shows us that the image passed in the model is a dog by indicating cat to 0 and dog to 1.

