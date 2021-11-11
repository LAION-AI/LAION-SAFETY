# LAION-SAFETY
A open toolbox for NSFW &amp; toxicity detection

# Overview
We present a NSFW image-text-pair classifcation ensemble, which consists of an image classifier ( based on EfficientNet V2, B2 260x260, https://github.com/google/automl/tree/master/efficientnetv2 ) combined with Detoxify ( https://github.com/unitaryai/detoxify ), an existing language model for toxicity detection.

The image classifier had been trained on 682550 images from the 5 classes "Drawing" (39026), "Hentai" (28134), "Neutral" (369507), "Porn" (207969) & "Sexy" (37914).

To evaluate the performance of the image classifier together with & without additional information from Detoxify, we created a manually  inspected test set that consists of 4900 samples, that contains images & their captions.

![image](https://cdn.discordapp.com/attachments/893170386030694460/908071613520560160/unknown.png)

To use our 5 class image classifier as a binary SFW - NSFW classifier, we consider images from the classes "Drawing" & "Neutral" as SFW and "Hentai", "Porn" & "Sexy" as NSFW.

--> Our image classifier predicts 96,45 % of the true NSFW correctly as NSFW and discards 7,96 % of the SFW images incorrectly as NSFW.


False negatives: 3,55% 

False positives: 7,96%


We compare our model with the best NSFW classifier from the Github user Gantman (https://github.com/GantMan/nsfw_model , Inception V3, Keras 299x299), to our knowledge the best openly available NSFW classifier at the time of writing:

![image](https://cdn.discordapp.com/attachments/893170386030694460/905489671654613102/unknown.png)


False negatives: 5,90%

False positives: 7,52%
 
--> Our image classifier predicts ~ 2 % less false negatives, at the cost of predicting ~0,5% more SFW pictures as NSFW. 
Because reducing the percentage of false negatives is more important in most contexts, the slightly increased percentage of false positives should be acceptable in most use cases.


To leverage the information from the image captions, we add the sum of Detoxify's "toxicity" & "sexual_explicity" scores to the softmax scores of the image classifier before determining the category with the highest score.

This ensemble archives the following performance:

![image](https://cdn.discordapp.com/attachments/893170386030694460/908072103465599026/unknown.png)


False negatives: 2,22% 

False positives: 5,33%

--> This ensemble predicts 1,3 % less false negatives & 2,6 % less false positives than our image classifier alone.


# Inference



# Training



# Disclaimer
Even though this is obvious, we explicitly state here that the predictions made by our image classifier & its ensemble with Detoxify are not 100% correct & that everyone who applies them has to take the full responsibilty for this application. 
