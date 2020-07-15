☀ [1. Introduction to Amazon Rekognition]()<br>

☀ [2. Introduction to AWS DeepLens]()<br>

☀ [3. Hands-on Rekognition — Automated Video Editing]()<br>

☀ [4. Deep Dive on Amazon Rekognition]()<br><br>

# 1. Introduction to Amazon Rekognition
<br>

## Intro

* Image and facial recognition service
* Deep learning based
* Simple API to analyze images
* Fully managed

**Rekongition** is one of AWS's offerings in AI<br>

Allows you to easily analyze images<br>
Your applications can make calls to Rekognition allowing you
to `recognize faces, landscapes, and even the mood of a person`<br><br>


## Benefits of Rekognition

1. Easy to Use
	--> Image recognition at the push of a button
	--> No expertise required
	--> APIs available, allowing applications easy use of AI
  
2. Extremely low cost of usage

3. Scalable<br><br>

As a managed service, Amazon will handle auto-scaling of
Rekognition allowing you to potentially send thousands of
images an hour for analysis through Rekognition<br><br>

## Rekognition Key Features

1. Object and scene detection
2. Facial analysis
3. Face comparison
4. Facial recognition
5. Confidence Scores on Processed Images<br>

The engine is able to determine mood, the no. of faces, as well
as differentiating faces from each other.<br>

All processed images give you a confidence score giving you the
ability to drop low confidence processes or perhaps flag them
for secondary or even manual review<br><br>

## Use Cases

### A. Searchable Image Library

1. (Mobile App) User captures an image for the property listing

2. (S3) The mobile app uploads the new image to Amazon S3

3. (AWS Lambda) A Lambda function is triggered and calls Amazon Rekognition

4. (Detect Objects & Scenes) Rekognition retrieves the image from
Amazon S3 and returns labels for detected
property and amenities

5. (Elasticsearch) AWS Lambda also pushes the labels and confidence score into Amazon Elasticsearch Service
 
6. (Property Search) Other users can search for and view the property<br><br>


--> Can be useful for many cases such as a real estate agent looking to take a picture of properties<br>

--> This makes searching for keywords such as "garage" or "yard" far easier for any real estate application<br><br>



### B. Image Moderation

1. Users upload pictures to S3 Bucket

2. S3 Bucket Object Created Events trigger Lambda functions

3. Lambda then sends data to Rekognition Image Moderation

4. If no inappropriate content is detected in image, the picture is posted to end users 

5. If inappropriate content is detected in image, it's sent for manual review and the user is notified if the image gets rejected.<br>
   After the image is approved to be appropriate, it's posted to end users<br><br>


--> This functionality can greatly assist foreign moderators ensuring that inappropriate content is not posted<br>

--> If confidence scores aren't high enough, it sends images for manual review ensuring that only approved content is shown.<br><br>

### C. Sentiment Analysis

1. (Live Image) User captures an image for the property listing

2. (In-Store Camera) Uses in-store cameras to capture live images of shoppers

3. Sends this to the application

4. (Analyze Faces) Amazon Rekognition analyzes the image and returns facial attributes detected, which include emotion and
   demographic details

5. (S3) Data is sent via Amazon S3 en-route to Amazon Redshift

6. (Redshift) Periodically copy data files to Amazon Redshift

7. (Quicksight) Regular analysis to identify demographic activity
                and in-store sentiment over time

8. Marketing Reports<br><br>


--> Has many use cases ==> From retailers wanting to track the effectiveness of sales
    or demographic info of the shoppers to a company that just wants
    to ensure that their workforce is happy.<br>

--> Deduced info can be put into marketing reports to track the
    effectiveness of your ongoing sales, upgrades to the stores, or
    just changes in your overall merchandise over time<br><br>


In summary, Rekognition is a great tool in our ML offerings<br>

Rekognition gives you the ability to run multiple forms of
analysis on your images and returns labels with confidence
scores to ensure the analysis is accurate.<br><br>

***

# 2. Introduction to AWS DeepLens
<br>

## In this course

* A look at the device
* AWS DeepLens architecture and workflow
* Sample project templates
* Demo == Building an object detection application<br><br>

## Intro to AWS DeepLens

1. `Wireless-enabled camera and development platform` integrated
   with the AWS Cloud

2. Offers the latest deep learning technology to develop computer
   vision applications<br>

--> Lets you use the latest AI tools and tech to develop
    computer vision applications based on deep learning models<br><br>


## The power behind the camera

--> What makes AWS DeepLens stand apart is the on-board **Intel Atom processor
    accelerator** that is capable of delivering 100 gigaflops
of compute, which means it can run `100 billion operations per
second`<br><br>

## The AWS DeepLens ecosystem

To create and run an AWS DeepLens project, you typically use
**Amazon SageMaker, AWS Lambda and AWS Greengrass**<br>

1. Use Amazon SageMaker to train and validate a custom model
   or import a pre-trained model

2. AWS Lambda functions in DeepLens perform 3 important
   operations == `pre-processing, capturing inference, and
   displaying output`<br><br>

Once a project is deployed to DeepLens, the model and the
Lambda function can run locally on the device<br>

3. AWS DeepLens creates a computer vision application project
   that consists of the model and inference Lambda function

4. AWS Greengrass can deploy the project and a Lambda runtime
   to your AWS DeepLens, as well as the software or configuration
   updates<br><br>


First, when turned on, AWS DeepLens captures a video stream.<br>

It produces `two output streams`<br>
a. **Device stream** — video stream passed through w/o processing<br>
b. **Project stream** — contains the results of the model's processing
		     video frames<br>

From there, the inference Lambda function receives unprocessed
video frames and passes them to the project's deep learning
model for processing.<br>

Finally, the inference Lambda function receives the processed
frames back from the model and passes them on in the Project
stream<br><br>


AWS DeepLens is well integrated with other AWS services<br>

For instance, projects deployed to DeepLens are securely
transferred using **AWS Greengrass**<br>

The output of AWS DeepLens, when connected to the Internet
can be sent back to the console via **AWS IoT** and **Amazon Kinesis
Video Streams**<br><br>

## Deep learning project templates

AWS DeepLens offers ready to deploy sample projects, which
include a pre-trained CNN model and the corresponding inference
function

DeepLens offers `7 sample project templates ready-to-use —`<br>
* Object detection
* Hot dog / not hot dog
* Cat and dog
* Artistic style transfer
* Activity recognition
* Face detection
* Head pose detection<br>

With these sample projects, you can get started with ML in
< 10 minutes<br>

These templated projects can be edited and extended
e.g. you could use the Object Detection project template to
recognize when your dog is sitting on your couch and have the
application send you an SMS to notify you of this event<br><br>

## Create your own custom model

Using supported DL frameworks, such as MXNet, TensorFlow, Caffe<br>

Of course, you can also create, train, and deploy your own
custom model to AWS DeepLens<br><br>

## Demo overview — Object Detection

The object detection project uses the `single shot multi-box
detective framework` to detect objects with a **pre-trained
ResNet50 network**<br>

The network has been trained on the Pascal VOC dataset and
can recognize 20 different objects<br>

The model takes the video stream from your AWS DeepLens as input
and labels the objects that it identifies<br>

It's a pre-trained optimized model that is ready to be deployed
to your AWS DeepLens<br>

After deploying it, you can review the objects DeepLens
recognizes around you through the console.<br><br>

***

# 3. Hands on Rekognition — Automated Video Editing
<br>

## Agenda

a. **Still Images**<br>
	Take a photo, send it to Rekognition & recognize faces
	and even objects within the picture<br>

b. **Video files**<br>
	Upload a video to S3, send it to Rekognition, and you
	can say "Hey, I found these various people within the
	video."<br>

c. **Streaming Video**<br>
	Send streaming video to Rekognition and it can detect
	objects in that streaming video and you can react to it
	in real-time<br><br>


### I. Still Images

> Rekognizing faces in image files, slightly real-time


Sample code ==> Captures an image from built-in webcam, shrinks it<br>
		(Rekognition only needs about 100 pixels to be able to detect a face)
		, sends it to Rekognition, calls the `detect_faces` command<br>

`detect_faces` can either point to an image in S3, or you can send it the
actual image from your code<br>

Then Rekognition comes back with a whole load of info about
the faces it saw in the image<br>

```
while(True):
    # Capture frame
    frame = capture.read()...

    # Resize image for faster rekognition
    image = capture.resize(frame, 0.15, ...)

    # Detect faces in image
    faces = rekognition.detect_faces(Image = {'Bytes':image}, ...)

    # Draw rectangle around faces
    for face in faces['FaceDetails']:
      is_smiling = face['Smile']['Value']
      draw_rectangle(image, face['BoundingBox'], is_smiling)
```
<br>

### Attributes returned by detect_faces()

Bounding box used to draw rectangle around each of the faces

```
{
  "FaceDetails": [ {
    "AgeRange": {"High"..."Low"...},
    "Beard": {...},
    "BoundingBox": {"Height".."Left".."Top".."Width"..},
    "Eyeglasses": {...},
    "EyesOpen": {...},
    "Gender": {...},
    "MouthOpen": {...},
    "Mustache": {...},
    "Smile": {...},
    "Sunglasses": {...}
] }
```
<br>

### Part A — Rekognize face within a picture

Do this by building something called a **"face collection"**<br><br>


### Loop — Find face in collection

Captures the picture once again, sends it across to Rekognition
and calls the `search_faces_by_image` command.<br>

It looks through a face collection.<br>
Rekognition then comes back with which face it saw, the name of that face
Then simply display it on screen who it said it was<br>

If multiple faces in picture, it detects the largest face<br>

```
while(True):

  # Detect faces in image
  faces = rekognition.search_faces_by_image(
		CollectionId = '...', Image = {'Bytes':image}, ...)

  # Show name and confidence
  for matches in faces['FaceMatches']:
    if 'ExternalImageId' in matches['Face'].keys():
      name = matches['Face']['ExternalImageId']
      confidence = matches['Face']['Confidence']

      cv2.putText(...)
```
<br>

### Attributes returned by search_faces_by_image()

```

{'SearchedFaceBoundingBox': {...},
 'SearchedFaceConfidence': 99.9997787475586,
 'FaceMatches': [
    {'Face': {'BoundingBox': {...},
	      'ExternalImageId': 'John',
	      'Confidence': 99.99739837646484
             }
          }
     ]
} 
```
<br>

### Part B — Rekognize text within the image

It comes back in a JSON object and shows me the text that
was there and it shows where it was found on the screen<br><br>


### Attributes returned by detect_text()

```
{u'TextDetections': [
      {'Geometry': {'BoundingBox': ...},
       'Confidence': 99.93452453613281,
       'DetectedText': 'THANKS',
       'Type': 'LINE'
      },
```

```
Also: 'Type': 'WORD'
```
<br>

Rekognition not only returns each individual word, it can
return a line of text<br>

If you're not interested in where the text is placed, and
everything that it's saying instead, it will return that
info to you as well<br><br>

### Functions for still images

Function | Description
-------- | -----------
**CompareFaces** | Match face between images
**DetectFaces** | Find up to 100 faces in an image
**DetectLabels** | Find real-world entities in an image
**DetectModerationLabels** | Detect adult content
**DetectText** | Find machine-readable text
**IndexFaces** | Add 100 faces in an image to a collection
**RecognizeCelebrities** | Identify up to 100 celebrities in an image
**SearchFaces** | Find a face ID in a face collection
**SearchFacesByImage** | Find largest face in a face collection
<br>

## II. Hands-On Rekognition Video
> Let's get Hands-On!

Detect faces in a video<br>

We have a very large video file with lots of different people
in it. <br>
We'd like to take that large video file and automatically
create a new video which is just showing one particular person<br>

To do this, Rekognition will have to go through the video,
find each person, output the info, and let us splice it
together again<br><br>


1. First thing I did is I captured a picture of each of the
people that were in the video, and used the create_collection
command which created a face collection

2. Then I had a picture of each person and I added it into the
collection using the `index_faces` command, which points to a picture
in S3, and I give it a name. Rekognition will associate that
external image ID (name) with that picture<br>

Rekognition does not store that image — it looks at the image,
finds the face in the image and it looks at the attributes of
the face (where are the eyes, nose, mouth, etc.) and it builds
a mathematical vector of the face and that is the only thing
that's stored in Rekognition<br>

Once Rekognition has those pictures, I can tell it to analyze
the video, so I use the `start_face_search` command which says
"Please go off to S3, grab this video for me and compare it to
the face collection called *'trainers'* "<br>


### Create a Collection & Load Faces

```
$ aws rekognition create-collection
--collection-id trainers

$ aws rekognition index-faces
--collection-id trainers
--image "S3Object={Bucket=hands-on-rekognition, Name=John.jpg}"
--external-image-id John
```
<br>

### Search for faces

```
$ aws rekognition start-face-search
--video "S3Object={Bucket=..., Name=trainers.mp4}"
--collection-id trainers

$ aws rekognition get-face-search --job-id...
```
<br>

`start-face-search` command tells Rekognition to start processing
the video but it can take a long time so it comes back with a
job ID, which can be used later to get output for a particular
job<br>

`get-face-search` command retrieves the result of the search and
outputs a lot of info on the screen<br>

It returns, for various different timestamps, who it has seen<br><br>


### get-face-search returns Timestamps and Faces

```
{ "Persons": [
  {
    "Timestamp": 7360,
    "FaceMatches": [
      {
        "Face": {
          "ExternalImageId": "John",
          "Confidence": 99.99750518798828..}
      }
    ],
  },
  {
    "Timestamp": 7560,
 ```
<br>

**CHALLENGE** == Take all of this info about where people appear
in the video and output another video just showing one particular
person<br>

We'll use **Amazon Elastic Transcoder** which is a service that
can transcode video files;
one of the capabilities it has is "clip stitching" == "go to
this video at this particular timestamp, grab x seconds of video
and keep doing it.." ==> splice them together into one single video<br>

Challenge in this case -- Rekognition outputs the timestamps
when it sees a particular person, while Elastic Transcoder
just wants to know when to begin and how long to record.<br>

So I've got to convert this format of timestamps into individual
scenes<br>

**--> Writing Python program**<br>
"John appears at these particular timestamps, let's convert it
into scenes.<br>
If he disappears for more than a second, end that scene and start it again
when he reappears OR if he appears for < 1 second, let's skip
over that scene because we don't want the video to be too jerky,
cutting in and cutting out"<br><br>


Blog Post == ["Automated video editing with YOU as the star!"](https://aws.amazon.com/blogs/machine-learning/automated-video-editing-with-you-as-the-star/)<br><br>


### Functions for stored videos

Function | Description
-------- | -----------
**StartCelebrityRecognition** | Recognize celebrities in a stored video
**StartContentModeration** | Detect adult content in a stored video
**StartFaceDetection** | Detect faces in a stored video
**StartFaceSearch** | Search for faces in a collection that match faces of persons detected in a stored video
**StartLabelDetection** | Detect real-world entities in a video
**StartPersonTracking** | Track persons in a stored video
<br>

## III. Rekognition Streaming Video

>	Face Detection on streaming video

We actually stream video to Amazon Kinesis Video, not Rekognition.<br>

Kinesis captures every video frame, sends that frame to Rekognition,
Rekognition then performs the same analysis we've seen so far<br><br>


### Demo

Set up a video camera showing people's faces, pass it through
Amazon Video Rekognition using Amazon Kinesis Video, use it to
trigger a light<br>

E.g. turn lights of the house on when you step in front of
camera installed on door<br><br>


### Architecture

First step is we set up a live video stream into Amazon Kinesis
Video Streams. <br>
It then sent each frame to Amazon Rekognition Video which
analyzed the frame and looked for certain people's faces.<br>

It compared those faces against the previously created face
collection. It then outputs that info in an Amazon Kinesis
Stream, that stream has all that JSON info<br>

Then we used Amazon Kinesis Analytics (it looks at streaming
info coming through the Amazon Kinesis stream and we said
"look for the face of XYZ being successfully recognized in
the video image"<br>

When his face was detected, we triggered an AWS Lambda function
that went out and asked IoT thing to turn on the light bulb or
open the front door<br><br>

***

# 4. Deep Dive on Amazon Rekognition
<br>

## Agenda
* Computer Vision
* Amazon Rekognition
* Amazon Rekognition API
* Use cases and Reference Architectures
* Best Practices
<br>

Goal is using ML models to be able to detect what are the
objects, what's the scene, what are the activities in these
images and be able to make sense of this content<br>

To detect different objects, scenes and activities, generally
you have to build ML models.<br>
To do that you need lots and lots of data<br>

e.g. to detect cats and dogs in images, you need lots of images
containing cats and dogs<br><br>


## Amazon ML Stack

I. **Application Services**<br>
Rekognition<br>
Transcribe<br>
Translate<br>
Polly<br>
Comprehend<br>
Lex<br><br>

II. **Platform Services**<br>

SageMaker ==> Build, train, and host ML models<br><br>

III. **Frameworks & Interfaces**<br>
AWS Deep Learning AMIs 
(Caffe2, CNTK, Apache MXNet, PyTorch, TensorFlow, Chainer,
Keras, Gluon)<br><br>

IV. **Education**<br>
AWS DeepLens<br><br>


## Amazon Rekognition Image

Rekognition is a DL-based image and video analysis service<br>

**[Images]**<br>
* Object & Scene Detection
* Facial Analysis
* Face Recognition
* Celebrity Recognition
* Unsafe Image Detection
* Text in Image<br><br>

**[Videos]**<br>
--> can detect activities for which usually you need more than
    just one frame; for activities where you have to look at a
    couple of frames together to make sense<br>
    
* Object & Activity Detection
* Pathing ==> Track the path different people took in the video
* Face Detection & Recognition
* Celebrity Recognition
* Unsafe Video Detection
* Real-time Live Stream<br><br>


### A. Object & Scene Detection

>	Identify objects and scenes with confidence scores

--> ...makes it easy for you to add features that search, filter,
    and curate large image libraries<br>

Take an image, send it to Rekognition<br>
Rekognition looks at the image, detects different objects,
what is in the scene and return us a list of labels<br>

Earlier, in media libraries containing unstructured content such
as images and videos, every single image or video had to be
tagged to make the library more searchable, also to be able to
easily find content<br>

Rekognition makes this easier<br><br>


### B. Facial Analysis

>	Analyze facial characteristics in multiple dimensions

* Demographic Data ==> Age Range, Gender
* Facial Landmarks ==> Nose, LeftPupil, RightPupil, BoundingBox, etc.
* Image Quality ==> Brightness, Sharpness
* Emotion Expressed ==> Happy, Surprised, etc.
* General Attributes ==> Smile, EyesOpen, Beard, Mustache
* Facial Pose ==> Pitch, Roll, Yaw<br><br>


### C. Facial Detection and Analysis

*	Support for up to 100 faces<br><br>


### D. Face Comparison

>	Measure the likelihood that faces are of the same person <br><br>


### E. Facial Recognition

> Identify a person in a photo or video using your private
	repository of face images

Index --> Collection --> Search<br><br>

### F. Unsafe Content Detection

>	Detect explicit and suggestive content<br><br>


**Explicit and Suggestive Content Labels**<br>

Top-Level Category | Second-Level Category
------------------ | ---------------------
Explicit / Nudity | Nudity
Explicit / Nudity	| Graphic Male Nudity
Explicit / Nudity	| Graphic Female Nudity
Explicit / Nudity | Sexual Activity
Explicit / Nudity | Partial Nudity
Suggestive | Female Swimwear or Underwear
Suggestive | Male Swimwear or Underwear
Suggestive | Revealing Clothes
<br>


### G. Celebrity Recognition
>	Recognize thousands of famous individuals

### H. Detecting Text
>	Detect and recognize text from images

### I. Pathing
>	Capture the path of people in the scene
<br>

## Amazon Rekognition API

**I. Object & Scene Detection -- Image API**<br>

(DetectLabels) == to get different labels for an image<br>

```
{
  "Image": {
      "Bytes": blob,
      "S3Object": {
          "Bucket": "string",
          "Name": "string",
          "Version": "string"
      }
   },
  "MaxLabels": number,  // how many labels you want returned
  "MinConfidence": number  // only returns labels having confidence level >= MinConfidence
}
```
<br>  
  
Response shown below<br>

```
{ 
  "Labels": [
      {
        "Confidence": number,
        "Name": "string"
      }
   ],
  "OrientationCorrection": "string"
}
```
<br>


**II. Object & Scene Detection -- Video API**<br>

* Asynchronous API<br>

(StartLabelDetection) == starts job to analyze video<br>

```
{
  "ClientRequestToken": "string",
  "JobTag": "string",
  "MinConfidence": number,
  "NotificationChannel": {
      "RoleArn": "string",
      "SNSTopicArn": "string"  // Job calls SNS topic to notify job completion
   },
  "Video": {
      "S3Object": {
          "Bucket": "string",
          "Name": "string",
          "Version": "string"
      }
   }
}
```

Returns a JobId<br><br>


**(GetLabelDetection)**<br>

```
{ 
  "JobStatus": string,
  "StatusMessage": string,   // tells whether job is complete or not
  "VideoMetadata": {
      "Format": string,
      "Codec": string,
      "DurationMillis": number,
      "FrameRate": float,
      "FrameWidth": number, 
      "FrameHeight": number
  },
  "NextToken": string,
  "Labels": [
      {
        "Timestamp": number,
        "Label": 
            {
              "Name": string,
              "Confidence": float
            }
      }
  ],
```
<br>

### Rekognition APIs — Overview

Rekognition's computer vision API operations can be grouped
into [Non-storage and Storage-based] API operations<br>


**Non-storage API Operations**<br>
DetectLabels<br>
DetectModerationLabels<br>
CompareFaces<br>
DetectFaces<br>
RecognizeCelebrities<br>
GetCelebrityInfo<br><br>

**Storage-based API Operations**<br>
IndexFaces<br>
DeleteFaces<br>
ListFaces<br>
SearchFaces<br>
SearchFacesByImage<br><br>


## Use Cases

### I. Celebrity Guests at the Royal Wedding

--> "Who's Who Live" function by SkyNews<br>
--> Celebrity guests identified on live stream<br>
--> On Screen captions of relations to the royal couple<br><br>

 
### II. Sports media tagging

--> Player recognition<br>
--> Motion path tracking<br>
--> Objects, activities, and event detection<br><br>


### III. Combatting Human Trafficking

>	ML supporting law enforcement and victim rescue

--> ML and analytics platforms for law enforcement<br>
--> Match photos of exploited children to those on the dark web<br>
--> Reduced the time and effort to identify and rescue victims<br><br>

Marinus Analytics, International Centre for Missing & Exploited Children, THORN<br><br>


### IV. Analyze User Generated Content

--> Ensure uploaded content is appropriate and moderated<br><br>

## AWS solutions

I.

1. Take a live photo
2. Submit the photo to some application
3. S3 stores the photo
4. Lambda function's triggered
5. Lambda function calls AWS Step Functions (service to build powerful workflows)<br>
Step Functions can then call different Lambda functions e.g.<br>
DetectFaces<br>
RecognizeCelebrities<br>
DetectModerationLabels   (find if there's any explicit or suggestive content)<br>
SearchFaces, etc.<br>
You can also match for any Blacklisted Images (stored on S3)<br>

6. Metadata can be stored on DynamoDB 
7. Make it searchable using AWS Elasticsearch Service<br><br>


II. **Image Moderation Chatbot**

1. User posts a message containing an image to a chat app channel that's monitored by a chatbot

2. The chat app posts the event to an Amazon API Gateway API for the chatbot

3. The chatbot validates the event. This event triggers an AWS Lambda function that downloads the image

4. Amazon Rekognition's image recognition feature checks the image for suggestive or explicit content

5. The chat app API deletes an image containing explicit / suggestive content from the chat channel

6. The chatbot uses the chat app API to post a message to the chat channel detailing deletion of the image<br><br>


https://github.com/aws-samples/lambda-refarch-image-moderation-chatbot<br><br>


**III. How long do people have to wait in the line?**<br>

You can have a camera pointed at the line and using Rekognition,
you can know how many people are in the line ==> how much time<br>

1. Live picture taken by camera

2. Call API Gateway

3. Pass that image to Lambda

4. Lambda can store that image in S3, call Rekognition to analyze image
to find how many people are there

5. Store metadata in DynamoDB<br><br>



**IV. Live Demographic Analysis**<br>

Sentiment analysis of the people, age, gender, etc.<br><br>


**V. Real-time shopping store heat map**<br>

Where there are more people<br>
Which section of the store has more footfall<br><br>

1. Live picture taken by camera

2. Pass them to API Gateway

3. Calls Lambda function

4. Lambda stores images to S3, call Rekognition to detect faces,
   do sentiment analysis, etc.

5. Store metadata and analysis to DynamoDB or Redshift

6. Use QuickSight to do analysis<br><br>



## Best Practices — Interfacing with Rekognition

1. **Image format** <br>
	PNG or JPEG

2. **Max. image size**<br>
	Amazon S3 = 15 MB<br>
	API calls = 5 MB (base64 encoded)<br>
 
3. **Video formats**<br>
	MP4, MOV

4. **Video Codec** == H264<br>

5. **Max. video size** == 8 GB

6. **Image resolution** == min. 80 px

7. Collections are for faces (not cats, cartoons..)

8. Max. no. of faces in a single face collection == 20 million<br>
	Latency is still < 1 sec<br>

9. Max matching faces the search API returns == 4096

10. Keep your images to re-index collection for future versions
    because Rekognition doesn't store images

11. IndexFaces detects largest 100 faces in the input image and
    adds them to the specified collection 

12. SearchFacesByImage detects the largest face in the image, and then
    searches the specified collection for matching faces<br>

    DetectFaces will return all faces in image<br>

13. Avoid false positives by not indexing blurry or low quality
    images<br><br>
 


If you have videos having sizes > 8 GB, you can use Elemental
Transcoder or other services to break them into smaller pieces
and then feed them one by one<br><br>

### Media Analysis Solution

https://aws.amazon.com/answers/media-entertainment/media-analysis-solution<br>

1. CloudFormation template which creates a bunch of resources
   including a Lambda function, Step Functions, S3 Buckets

2. Gives you a nice UI where you can upload some images and videos
   & quickly get to see all the metadata that different Rekognition APIs
   have returned

3. Creates an Elasticsearch cluster which gives you an option to see how
   you can build a media library and then quickly be able to
   search by using the different metadata that you get from Rekognition

***
