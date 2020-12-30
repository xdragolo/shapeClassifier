#Task: Recognize pictures of X and O
1. obtain testing data (picture of 10x10 px from internet, own photos and/or own pictures)
2. create model using NN (Neural Network)
3. train NN
4. evaluate NN
5. create stand alone application which is using camera as input and classification of X and O as output

## Project structure
1. Figures (directory containing learning curves pics)
2. Pictures (directory for picture made by app)
3. appClassifyPicture.py (script that run pc camera, on pressing space button makes photo and then classify made photo)
4. imagePreprocessing (script containing function for picture preprocessing)
5. main.py (script for model fitting and analysis)
6. model.py (function returning NN model)


## Dataset
I used two groups (stars and circle) from https://www.kaggle.com/smeschke/four-shapes

## Aproach description
*Image preprocessing*
- Dataset contains pictures in png format 200x200px, I resize all these pictures to 10x10px, so every pic is represented by vetor with 300 numbers (10x10x3 (RGB))

*Model*
- I created a simple base model of NN (one hidden layer), which perform too well to be true:
    * I realized that data are too easy to interpret (contains only black and white spots) and unrealistic for use-case (classify pictures from pc camera)
    * I decided to add to the training dataset image noise. All numbers in the vector are multiplied by a random number between 0 and 1. It will affect all spots that are not black.
- Even with adjusted data I was able to reach accuracy on testing data over 90 %.
- I try to classify pictures made by pc camera, unfortunately unsuccessfully. I tried:
    * Convert the picture into black and white colors
    * Increase contrast on black and white picture to make them similar to training pictures
- I assume that the biggest obstacle for classifying pictures made by pc camera is size reduction to 10x10px. The resulting picture contains too much blur, and my classifier always perceives that as a star shape.

## Possible improvements:
- use different training data - real pictures of shapes
- use different size of pictures
