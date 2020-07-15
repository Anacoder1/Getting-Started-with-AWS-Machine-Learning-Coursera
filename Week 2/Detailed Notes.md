# The Machine Learning Process
<br>

## Agenda

* Define key terms and concepts
* Explain the ML pipeline
* Discuss using the ML pipeline to solve a real-world problem<br><br>

Amazon built a system, they used ML to improve the whole routing
system<br><br>

### ML Pipeline

1. ML Problem Framing
2. Data Collection & Integration
3. Data Preparation
4. Data Visualization & Analysis
5. Feature Selection & Engineering
6. Model Training
7. Model Evaluation
8. Prediction<br><br>

## 1. ML Problem Framing

Business Problem — How to route customer calls successfully?<br>

ML Problem — Predict the agent skills<br><br>

ML is a subset of AI<br>
ML uses data to train the model which will be used to make
predictions<br>

Predictions can be made from huge datasets<br>
Strength lies in its ability to extract hidden patterns, structures
from this data<br>


#### Three ML problems

1. Binary — Two groups
2. Multi-class — > 2 groups
3. Regression — Continuous values (e.g. predicting price of company stock)<br><br>


### Questions asked

* What exactly did these customer service agents' skills
represent?

* How much overlap were there b/w the skills?

* Are they similar enough to possibly combine them?

* What happens when the customer's routed to an agent with
wrong skills  

* Did the agent stand a chance of possibly answering the
question anyway?<br><br>


More questions you ask during this discovery stage, the more
inputs the domain experts give you ==> Better your model will
be<br><br>


## 2. Data Collection & Integration

> Collect and integrate the data that's relevant to your problem
<br>

Data is everywhere, can be collected from multiple sources
like Internet, databases, other types of storage<br>

Data's going to be `noisy`<br>
Data can be `possibly incomplete, even irrelevant`<br>

Wherever it comes from, it'll need to be compile, get
integrated. Most importantly, you have to clean the data<br>

* No matter what type of data you have, you're gonna need
to make sure that you've got the proper tools and the knowledge
to work with all different datatypes<br><br>


Features — inputs to the problem<br>

ML model's job during training is to `learn which of these
features are actually important to make the right prediction
for the future`<br>

Value is known --> **Label** (supervised learning)<br>
Value isn't known --> **Target** (unsupervised learning)<br><br>


In the call center example, the label was the *"skill an agent
needed to resolve the customer call"*<br>

Together, the features and labels make up a single "data point" = "observation"<br>

**Feature** — One column of data (input) such as order, device, Prime Membership<br>

**Label** — Determination (output), such as generalist, specialist<br>

**Dataset** — Bunch of observations<br><br>


Good data will contain a signal about the phenomenon you're
trying to model<br> 

* Representative features<br><br>


**A general rule of thumb is you should have atleast 10x the number of
data points as features**<br><br>


## 3. Data Preparation

The very first dataset isn't going to be enough for a good
prediction<br>

Important to understand what data you're missing so that you
can access it<br>

a) Take a small random sample of your data, really dig into it<br>


* Your job in the data prep phase is to `manually and critically
explore your data`<br>


Ask questions like this —<br>

* What features are there?
* Does it match your expectations?
* Is there enough info to make accurate predictions?<br>

If a human could look at a given data point and guess the correct label,
then an ML algo should be successful there too<br><br>


**Confirm all labels are relevant to the ML problem**<br><br>

* Are there any labels that you want to exclude from the biz
model for biz reasons?

* Are there any labels that aren't entirely accurate?<br><br>

## 4. Data Visualization & Analysis

It can be hard to understand your data without seeing the data<br>

More than manual analysis, need `programmatic analysis`<br>

`Visualization` helps you understand the relationships within
your dataset<br>
It leads to better features, better models<br>

* It can help unveil previously unseen patterns<br>
* It reveals corrupt data or outliers that you don't want, properties
that can be significant in your analysis<br><br>

Basic stats like pie charts, etc. can be powerful methods to
obtain quick feature and labeled summaries to understand them<br><br>

**Histograms**<br>

1. Effective visualizations for spotting outliers in data

2. You can delete outlier data or cap it (use data in a specific range)<br>

In regression problem, you can deal with outliers or even
missing data by just assigning a new value using **"imputation"**<br>

Imputation is going to make a `best guess as to what the value
actually should be`<br>
E.g. filling missing value with mean value of dataset<br><br>


**Scatter plots**<br>
1. Visualize the relationship b/w the features and the labels

* It's important to understand if there's a strong correlation
between features and labels<br>

2. Helps see correlation between features and labels<br><br>


* If you don't address noisy data, it's just going to hurt
your model's performance

* Noisy or missing data --> less accurate predictions<br><br>

### Choosing the right algo

1. Supervised
2. Unsupervised
3. Reinforcement
4. Deep Learning

Algo should make sense for your biz problem<br>

Choosing the right algo for job is another big step in this
part of the ML pipeline<br><br>


**A. Supervised algos**<br>

Focus is on `learning patterns by seeing the relationships
between variables and known outcomes`<br>

**Supervisor** (could be machine, human or other natural processes) shows the model the right answer<br><br>


After training is finished, a successful learning algo can
make the decisions on its own, you no longer need a teacher
to label things<br>

E.g. call center use case (data was historical customer data
that included the correct labels or the customer agent skills)<br>

* Need good training datasets, properly labeled observations
* This type of ML is only successful if the system we're trying
to model it after is already functioning and easy to observe<br><br>


**B. Unsupervised algos**<br>

Just the data, no labels or targets<br>

We don't know the variables, don't know the patterns
The `machine itself simply looks at the data and tries to create
labels all on its own`<br><br>


* **Clustering**

`Groups data points into different clusters based on similar
features`, in order to better understand the attributes of a
specific group or cluster<br>

To detect an unclassified category of fraud in the early phases,
like a sudden large order from an unknown user or a suspicious
shipping address, unsupervised algos group malicious actors
into a cluster and then analyze their connections to other accounts
w/o knowing the actual labels of the attack originally<br><br>

**C. Reinforcement Algos**<br>

Agent --> Action --> Environment --> State/Reward --> Agent ... [Loop]<br>

RL continually improves by mining feedback from previous
iterations

`Agent continually learns through trial and error as it
interacts with the environment`<br>

RL **broadly useful when the reward of a desired outcome is known
but the path to achieve it isn't**. <br>

That path requires a lot of trial and error to actually discover<br><br>

**D. Deep Learning**

It's a reinvention of ANNs<br>

Each neuron is activated when the sum of the input signals
into one neuron exceeds a particular threshold<br>

A single neuron isn't sufficient for practical classification
needs. Instead, we combine them into a fully connected set of
layers to produce ANNs — Multilayer perceptrons<br>

The computational power required to train such networks isn't cheap<br><br>


**A. CNN**<br>

One important breakthrough in deep learning was the invention
of CNNs, which are especially useful for image processing<br>

Main idea of CNNs is that `nearby pixels in the image are taken
into consideration instead of treating them as entirely
separate inputs`<br>

Special operation "Convolution" is applied to entire subsections
of the image<br>

If several conv layers are stacked one after the other, each
conv layer learns to recognize patterns that increase in
complexity as it moves through the layers<br><br>


**B. RNN**<br>

If we `take the output of a neuron and feed it as an input to
itself or to neurons of previous layers.`<br>

Instead of going just forwards, we feed backwards or maybe
into itself ==> RNN<br>

It's as if the neuron remembers the output from a previous
iteration, thus creating some kind of memory<br>

A more complex network is **LSTM**, it's commonly used for
`speech recognition or translation`<br><br>

## 5. Feature Selection & Engg

Select which features you want to use with your model<br>

* What you want to have is a minimal correlation among your
features, but you want to have the maximum correlation b/w
the features and the desired output<br>

Select features which correlate to your desired output<br> 

Part of selecting the best features includes recognizing when you've
got to engineer a feature<br>

**Feature engineering** = `process of manipulating your original
data into new and potentially a lot more useful features`<br>

Feature engg. is arguably the most critical and time-consuming
step of the ML pipeline<br>

It answers questions like 
1. Do the features I'm using make sense for what I want to
predict?
2. How can I systematically take what I've learned about my
features during the visualization process and encode that
info into new features?<br><br>


In call center use case, after visualizing location of customers
calling about tracking, you could engineer a feature for customers tracking packages
in specific cities<br>

This info might lead to same patterns you otherwise wouldn't
have seen before<br><br>


Feeding features into model training algo, it can only learn
from exactly what we show it<br>

E.g. in call center use case, "Date/Time of most recent order"
isn't a helpful feature compared to "Days since last purchase"<br><br>


E.g. in image classification model trained to identify cars,
feeding raw images of cars won't be helpful given that these
images are very complex combination of pixels.<br>

The raw data you're going to feed in doesn't include any
higher-level features such as edges, lines, circles, the
patterns that it can recognize <br>

So, during the feature engg stage, you can pre-process the data.
This will classify it, possibly get to more granular features,
that way, we can feed those features back into the model and get
better accuracy<br><br>

## 6. Model Training

First step you have to take while training the data, is you
**have to split it**<br>

Splitting the data allows you to ensure that you've got production
data that's similar to your training data that your `model will
as a result be more generalizable or applicable outside of
the training environent`<br><br>


Typically, you want to split your data into three sections —
`training data, dev data, and test data`<br>

**Training data** will include both the features and the labels, this
feeds into the algo you've selected to help produce your model<br>

The model is then used to make predictions over a developments
**(Dev) dataset**, which is where you'll likely notice things that
you'll want to tweak, tune and change<br>

Then when you're ready, you can actually run the **test dataset**,
which only includes features since you want the labels to be
what's predicted through the model<br><br>


**The performance you get with a test dataset is what you can
expect to see in production**<br>

The amount of data you have determines how ultimately you split
it up, but regardless, you'll want to train your model on as
much data as possible, knowing that you're gonna need to
reserve some of it for the dev and test phases<br><br>


**Make sure you randomize your data while splitting your data.**<br>
This is critical. This will help your model avoid bias.<br>

This is especially true with structured data, if your data
is coming in a specific order<br>

Popular randomization is simply `shuffling your data`<br>

`scikit-learn` tools will help to randomize and shuffle your data<br><br>


Randomizing and splitting your training data is a critical step
in the training process<br><br>


Common mistake people make is that they don't hold out testing
data, what they end up doing is simply testing on part of the
data they trained with, the training data.<br>

This doesn't generalize your model, it will actually lead to
either overfitting or underfitting<br><br>


**Overfitting** = where the `model learns the particulars of a dataset
              too well.`<br>
              
It's essentially *memorizing your training data
as opposed to learning the relationship b/w the features
and the labels* so the model can use what it learns in
those relationships to build patterns to apply to new data
in the future.<br>

In addition to simply randomizing the data, it's also very
important to collect as much relevant data as possible because
**underfitting** can occur `if you don't have enough features to
model the data properly`<br>

This can again prevent the model from properly generalizing the
data because it *doesn't have enough info to predict a right
answer*, to predict correct<br><br>


**Bias** = `The gap b/w predicted value and actual value`<br>

**Variance** = `How dispersed your predicted values are`<br><br>


**Low bias, Low variance model** ==><br> 
Everything's clustered tight and it's right there in the bull's eye<br>
I'm getting everything I predict in one area, there's not a lot of spread<br>

**High bias, Low variance model** ==> <br> 
I'm not getting everything that I want but at least I'm getting a
				   predictable series of responses. <br>
           It's a tight cluster, I'm just not
				   on the bull's eye<br>

**Low bias, High variance model** ==><br>
I'm on target as far as the center of spread goes, but the spread is
				  wide, it's all over the place<br>

**High bias, High variance model** ==>
I'm all over the place and I'm not on target<br><br>



`Ideal case` is a **Low bias, Low variance model.**<br>
Realistically though, there's a balancing act happening out here<br>

Bias and variance both contribute to errors, but you're
ultimately going for minimizing your prediction error,
not B or V specifically ==> **Bias-Variance tradeoff**<br><br>


**Underfitting** is where you've got `High bias, Low variance`<br>
	These models are overly simple, they can't see the
	underlying patterns in the data<br>

**Overfitting** == `Low bias, High variance`<br>
  These models are overly complex; while they can
	detect patterns in the training data, they're not
	accurate outside of the training data<br><br>


In testing and production, our model won't pay attention
to these other missing categories, it will skew the results
towards only the data that the model was actually trained on<br>

One technique that can be `used to combat both underfitting
and overfitting` = **hyperparameter tuning**<br><br>


A **"parameter"** is `internal of the model and it's something the
model can learn or estimate purely off of the data`<br>

E.g. the weight of an ANN or the coefficients in linear regression<br>

The model has to have parameters to make predictions and most
often, these aren't set by humans<br><br>


**"Hyperparameters"** are `external of the model and can't be estimated
from the data.`<br>

They're set by humans and typically you can't really know the
best value of the hyperparameter, but you can trial and error
and use that to get there<br>

Tuning hyperparameters will improve performance of ML model<br>

**The right hyperparameters have to be chosen for the right type
of problem**<br>

E.g the learning rate for training a NN<br><br>


**Different types of hyperparameters**<br>

a) Loss function<br>
b) Regularization<br>
c) Learning parameters<br>

Walking through this part of the process is one of the most
effective ways of improving your model's performance<br>

Ensure you take time to conduct H-tuning thoroughly<br><br>


The process of training an ML model involves providing your
algo with training data to learn from<br>
--> for Supervised learning, training data must contain both
    the features and the correct predictions (labels)<br>

The learning algo finds patterns in the training data that
maps the features to the label, so when you show the trained
model new inputs, it'll return accurately predicted labels<br>

Then you can use the ML model to get predictions on new
data for which you don't know the label<br>

E.g. predicting whether an email is spam or not spam<br><br>

## 7. Model Evaluation

After the initial phase of training your model is done, you'll
need to evaluate how accurate that model is `by using the development
data that you set aside and run it through the model`<br>

This is going to tell you how well you generalize the models<br>

The test data may be fed into the model for the most accurate
predictions<br><br>


While you're evaluating, you want to fit the data that
generalizes more towards unseen problems<br>

You should not fit the training data to obtain the max
accuracy (if you train your model to be too accurate, it
will be over-fit to that specific training data)<br><br>


One of the most effective ways to evaluate your model's accuracy,
precision, and ability to recall involves looking at a
**"confusion matrix"**<br>

Confusion matrix `analyzes the model and shows how many of the
data points were predicted correctly and incorrectly`<br><br>


**True Positive** == Predicted a 1, Got a 1<br>

**True Negative** == Predicted a 0, Got a 0<br>

**Class 1, Class 0 box** == Predicted a 1, Got a 0<br>

**Class 0, Class 1 box** == Predicted a 0, Got a 1<br><br>


**Accuracy** =  Degree of deviation from the truth =  (#Correct predictions) / (#Predictions)<br>

**Precision** = Ability to reproduce similar results<br>
	    What proportion of positive identifications was actually correct?<br>
          = TP / (TP + FP)<br><br>


Best practice of evaluation is running the model against a few
different algos<br>
Consider running it through a couple different algos within the
chosen algo category<br>

This will give you a better idea of how to get the best fit
and the best results for your model<br><br>


**Supervised algos**<br>
--> Decision trees (C)<br>
--> K-Nearest neighbors (C)<br>
--> Neural Networks<br>
--> Regression analysis<br>

**Unsupervised algos**<br>
--> K-means clustering<br>
--> Anomaly detection<br>
--> Neural Networks<br>

**Reinforcement learning**<br>
--> Q-learning<br>
--> SARSA<br><br>

## 8. Prediction

After you're satisfied with the model's predictions on unseen
data, it's time to deploy your model into production so it
can begin making your predictions<br>


One of the primary ML tools for building, training and deploying
models ==> **Amazon SageMaker**<br>

1. `Fully managed`, covers the entire end-to-end pipeline
2. The `build module` SageMaker provides a hosted environment
for you to work with your data, you can experiment with your
algos, you can visualize the output
3. `Train module` actually takes care of the model training and
tuning at high scale
4. `Deploy module`, designed to provide you a managed environment
for you to host, test models for inference, secure low latency, etc.

Additional tools in SageMaker help you label data, manage your
compute costs, take care of forecasting<br>


Remember to monitor your production data and retrain your model
if it's necessary because a newly deployed model needs to
reflect current production date, you don't want to get out of
date<br><br>

Since data distributions can drift over time, deploying a model
is not a one-time exercise, it's a continuous process<br>


Evaluating in a production setting is a little bit different.
Now you've got to have a very concrete success metric that you
can use to measure success<br>

In call center use case, our routing experiments were predicated
on the assumption that the ability to more accurately predict
skills would reduce the no. of transfers<br>

In production, we can put that assumption to test<br><br>


A ton goes into implementing a ML solution<br>

It's a process that most often takes several weeks or months<br><br>

***
