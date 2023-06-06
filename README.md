# birds

[Project Video](https://youtu.be/YWfjeZSYTmc)

[code](https://github.com/spoonk/birds/blob/main/455-final-project(1).ipynb)

# Problem Description

I chose to participate in CSE 455’s bi-annual bird classification competition. The goal of this competitions is to come up with a method for identifying the species of a bird. This type of problem typically involves training a neural network to make predictions, which is the approach that I decided to use. 

# Previous Work

Since I have done some work in Tensorflow previously, I decided to use Tensorflow. Tensorflow has many resources and tutorials as well pretrained models. The resources from Tensorflow that I used were [Transfer Learning and Fine-Tuning](https://www.tensorflow.org/tutorials/images/transfer_learning) as well as [Save and Load Models](https://www.tensorflow.org/tutorials/images/transfer_learning) . I also used several pretrained models from Tensorflow’s Keras API. The models that I used were MobileNetV2, VGG19, Resnet50, and Resnet152. These models were all pretrained on the imagenet dataset. 

# My approach

I decided to use transfer learning to adapt an existing, pretrained model to the task of bird species classification. I chose to use transfer learning for several reasons. Firstly, pretrained networks have already learned to detect objects in the real world. This means they can reason about parts of an image and have some baseline level of knowledge about images. Secondly, our data has 555 different species of birds, which is a very large number. I expected that a model would need to be complex and deep to be able to see the differences between images. I did not want to train a large model from scratch, so I went with transfer learning instead.

To do transfer learning, I took a pretrained model and ignored the final classification layer that the model used. This layer would have identified which class an image was from the imagenet dataset, but since I was using different data it was completely irrelevant. After removing that layer, I added a global average pooling layer to downsize the number of features, a dropout layer for regularization, and then two dense layers, the final of which was used to identify the class of an input image.

After deciding to do transfer learning, my next step was to determine which pretrained model I wanted to use. I decided to choose from MobileNetV2, VGG19, Resnet50, and Resnet152. After training each for 15 epochs, I saw that Resnet152 had the best performance so I decided to commit to it. I added data augmentation to my training pipeline then decided to train the dense layers that I added for 30 epochs. Then, I unfroze the top 100 layers of Resnet152 to fine-tune the model and trained for 20 more epochs.

I used this version of the model to make predictions, but after all that training I was only able to achieve a score of 0.64 on the Kaggle leaderboard :( I even tried using a EfficientNetV2 for transfer learning, but that did not help me achieve higher accuracy.

# Datasets

The dataset provided for the bird competition consisted of about 40,000 images of birds belonging to 555 different classes. The test set provided consisted of 10,000 unlabeled images and was used to generate a set of predictions to submit to Kaggle. In order to approximate a test set, I split the training data into a training set and a validation set. The validation gave me an estimate for how the model would perform when making predictions on data that it had never seen before.

# Results

My model was only able to achieve 64% accuracy on the Kaggle test set. In comparison to other teams on the leaderboard, this is pretty bad. However, considering that there were 555 different classes of birds, the fact that we can classify a bird correctly over 50% of the time is pretty cool. 

During training, the validation accuracy sometimes got above 80%. When I made my initial predictions for Kaggle, I made them with a model that achieved 82% accuracy on the validation set. These predictions ended up getting 62% accuracy on the test set.

During the fine-tuning phase of training, the validation accuracy dropped over time from around 80% to around 70%. The strange thing is that when I used this version of the model to make predictions on the test set, it got a score of 64% despite doing worse on the validation set during training. I discovered that the strange behavior of the model on the validation set was due to a bug in my code with how I was loading in training data. Despite fixing this bug, I still wasn’t able to achieve an accuracy over 64% despite tweaking 

# Discussion

## Problems encountered

The main problem that I encountered was that I couldn’t seem to achieve high accuracy despite training for quite a while. Although I understood the model and techniques such as regularization, I lacked the intuition for which hyperparameters to tweak and which modifications to make to the model to get better accuracy. I also didn’t take note which hyperparameters and models I had already tried and what their accuracies had been. This became an issue because I spent so much in between tweaking parameters to train the models that I forgot which combinations I had already tried. 

Another problem I encountered was that the validation set that I created didn’t give similar results to the test set, as discusses in the Results section. Thinking back, I am pretty sure I know why this is. I trained my model over multiple sessions where I would save the model’s weights, close the notebook, then on the next day I would re-run the notebook and load the model’s weights to resume training. The issue with this is that I used shuffling when loading the datasets, meaning the training and validation datasets did not include exactly the same data each time. When evaluating the model with the validation set, the model had already been trained on most of the images in the validation set because of the shuffling. After fixing this bug, I was able to work get meaningful metrics on subsequent training sessions.

## Next steps

If I had more time, I would create my own CNN and train it from scratch. I am curious to see whether the learned weights from training on imagenet placed an upper bound on the final accuracy of the model, because it was biased more towards images from the imagenet dataset. 

## How does my approach differ from others? Was that beneficial?

I decided to do transfer learning when I could have trained a model from scratch. I think that this did benefit me because I do not have much experience training models from scratch, and I would have doubted the architecture I chose for my custom model. Using transfer learning allowed me to trust the model architecture and focus on the hyperparameters. However, I wasn’t able to achieve a high accuracy using transfer learning so maybe the from-scratch approach would have been better
