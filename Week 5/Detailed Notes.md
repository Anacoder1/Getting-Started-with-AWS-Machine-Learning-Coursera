☀ [1. Introduction to Amazon SageMaker](https://github.com/Anacoder1/Getting-Started-with-AWS-Machine-Learning-Coursera/blob/master/Week%205/Detailed%20Notes.md#1-introduction-to-amazon-sagemaker)<br>

☀ [2. Introduction to Amazon SageMaker GroundTruth](https://github.com/Anacoder1/Getting-Started-with-AWS-Machine-Learning-Coursera/blob/master/Week%205/Detailed%20Notes.md#2-introduction-to-amazon-sagemaker-groundtruth)<br>

☀ [3. Introduction to Amazon SageMaker Neo](https://github.com/Anacoder1/Getting-Started-with-AWS-Machine-Learning-Coursera/blob/master/Week%205/Detailed%20Notes.md#3-introduction-to-amazon-sagemaker-neo)<br>

☀ [4. Automated Model Tuning using Amazon SageMaker](https://github.com/Anacoder1/Getting-Started-with-AWS-Machine-Learning-Coursera/blob/master/Week%205/Detailed%20Notes.md#4-automated-model-tuning-using-amazon-sagemaker)<br>

☀ [5. Amazon SageMaker — Object Detection on Images labeled with GroundTruth](https://github.com/Anacoder1/Getting-Started-with-AWS-Machine-Learning-Coursera/blob/master/Week%205/Detailed%20Notes.md#5-amazon-sagemaker--object-detection-on-images-labeled-with-groundtruth)<br>

☀ [6. Build a text classification model with Glue and SageMaker](https://github.com/Anacoder1/Getting-Started-with-AWS-Machine-Learning-Coursera/blob/master/Week%205/Detailed%20Notes.md#6-build-a-text-classification-model-with-glue-and-sagemaker)<br><br>


# 1. Introduction to Amazon SageMaker
<br>

## Intro to Amazon SageMaker

**Amazon SageMaker** is a `fully managed service that enables data scientists and
developers to quickly and easily build, train, and deploy ML models`<br><br>


## Amazon SageMaker Components

1. **Amazon SageMaker Notebooks service**<br>
   Hosted notebook instance

2. **Amazon SageMaker Training service**<br>
   Distributed, on-demand training environment

3. **Amazon SageMaker Hosting service**<br>
   Model hosting environment that is `elastic, scalable, secure, and reliable`<br>

Although using all components end-to-end brings you a seamless experience on
Amazon SageMaker, you have the flexibility to use any combination of those
components in order to fit your own workflow.<br><br>

## Zero Setup for Exploratory Data Analysis

Amazon SageMaker provides hosted Jupyter notebooks that require no setup.<br>

With a few clicks on the SageMaker console or through APIs, you can create
a fully managed notebook instance, which comes with preloaded data science
packages, such as **popular Python libraries, DL frameworks, Apache Spark,
and so on**.<br>

You can then start processing your datasets and developing your algorithms
immediately.<br><br>


If you want to use extra packages that are not preloaded, you can simply
`pip install` or `conda install` them, which will be persisted in that
notebook instance.<br>

Although you can certainly run training jobs in the hosted notebook instance,
many times, you may want to get access to more compute capacity, especially
for large datasets.<br>

In that case, you simply select the type and quantity of Amazon EC2
instances you need, and kick-off a training job.<br>


## High Performance, On-Demand

Amazon SageMaker then sets up the compute cluster, performs the training
job and tears down the cluster when the training job is finished.<br>

So you only **pay for what you use** and never worry about the underlying
infrastructure.<br><br>


In SageMaker, `model training is flexible.`<br>
You can certainly bring arbitrary algorithms, either open-sourced or
developed by yourself in the form of Docker images.<br>

Although Amazon SageMaker offers a range of `built-in high-performance
ML algorithms` that have been optimized for distributed training, making
it highly effective in training models against large datasets.<br>

For those who want to train their own neural networks, SageMaker
makes it super easy to directly submit your TensorFlow or Apache
MXNet scripts for distributed training;
you can use alternative DL frameworks as well by `packaging your own
Docker images and bringing them` to Amazon SageMaker<br><br>


## Easy Model Deployment to Amazon SageMaker Hosting

* **Auto-Scaling Inference APIs**
* **A/B Testing (more to come)**
* **Low Latency & High Throughput**
* **Bring Your Own Model**<br>

When you're ready to deploy a model to production, you can simply
indicate the compute resource requirements for hosting the model
and deploy it with just one click.<br>

A HTTPS endpoint will then be created to achieve `low latency, high
throughput inferences.`<br>

With Amazon SageMaker, you can swap the model behind an endpoint
without any downtime or even put multiple models behind an
endpoint for the purpose of A/B testing.<br><br>

## Demo

You can always use Amazon SageMaker console or APIs to build your
workflows, if you prefer to do that.<br>

Grant permissions to the notebook instance through an **IAM role**
to access necessary AWS resources from the notebook instance without
the need to provide an AWS credential.<br>

If you don't have an IAM role in place, SageMaker will automatically
create a role for you with your permission.<br><br>

For those wanting to access resources in VPCs, you can specify which
VPC you want to be able to connect from the notebook instance<br>

You can also secure your data in the notebook leveraging **KMS
(Key Management Service) encryption**<br>

Once the notebook's created, it will open the Jupyter dashboard.<br>
**Jupyter** is an `open-sourced web application that allows users to author
and execute code interactively.` It is very widely used by the data science
community.<br>

Example notebooks for all kinds of ML solutions are available for reference,
developed by subject matter experts across Amazon, and new examples are
added over time.<br>

One can make their own version of the example notebook by copying it.<br><br>

**XGBoost** is an extremely popular `open-source package for gradient boosted
trees`, which is widely used to build classification models.<br>

Amazon SageMaker offers XGBoost as a built-in algorithm so that customers
can access it more easily.<br><br>


In this notebook, we will `build a model to predict if a customer will
enroll for a term deposit at a bank after one or more outreach phone
costs.`<br>

We'll use a public dataset published by **UC Irvine** which contains information
about historical customer outreach and whether customers have subscribed to the
term deposit offered by a bank in Europe.<br>

1. We'll first set up some variables that are being used in this notebook
   and download the dataset from the internet.

2. After that, we'll perform some data exploration and transformation to
   prepare the data for training.

3. We'll then kick-off a training job in the Amazon SageMaker training
   environment.

4. After the training is done, we'll deploy the model to Amazon SageMaker
   Hosting and make inferences against the designated HTTPS endpoint<br><br>


Training data and validation data will be used in the training process.<br>
The test data will be used to evaluate model performance after it is
deployed.<br><br>


**[Deploying model into production]**<br>

We'll first create a model in Amazon SageMaker Hosting by specifying
the model artifacts' location as well as the inference image which contains
the inference code.<br>

Since XGBoost has been used, the inference image here is provided and
managed by SageMaker as well.<br>

If you were to bring your own model to hosting, you'd need to provide your
own inference image here.<br>

We'll then create an endpoint, but before that, we'll need to set up an
endpoint configuration first — This is to specify how many models we're
going to put behind an endpoint and the compute resources we'll need for
each of the model.<br><br>

***

# 2. Introduction to Amazon SageMaker GroundTruth
<br>

## What is GroundTruth?

**GroundTruth** is a `new capability of Amazon SageMaker that makes it easy for you
to efficiently and accurately label the datasets that are required for
training ML systems.`<br>

GroundTruth can automatically label some part of the training dataset and then
sends the rest to human workers for labeling.<br>

GroundTruth also uses innovative algorithms and user experience techniques to
improve the accuracy of the labeling that's sent to human workers.<br><br>

## The current method

Data labeling requires a lot of manual effort and is prone to errors<br>

* Often requires distributing task over a large number of human labelers,
  adding significant overhead and cost

* It leaves room for human error and bias

* It's a complex process to manage and can take months to complete<br><br>

## GroundTruth fixes these pain points

**GroundTruth Labeling Jobs**<br>

* Managed experience where customers can set up an end-to-end
  labeling job in just a few steps

  You simply need to provide a pointer to your data in S3

* Templates for common labeling tasks

  Where you only need to click a few choices and then provide
  some minimal instructions to the human workers to get your
  data labeled

* Built-in features to improve labeling accuracy

* Selection from multiple labeling workforces<br>

You select one of 3 workforce options. On completion of a labeling
job, GroundTruth will send the outputted labeled data to your S3
bucket.<br><br>


## Human labeling

You have 3 options when selecting a workforce to perform your labeling

1. **Public** crowdsourced workforce (Mechanical Turk)
2. Pre-approved **third-party vendors** (listed on AWS Marketplace)
3. Your own **internal workforce**<br>
     We host a labeling application on which you can onboard these
     workers<br><br>

## Better labeling accuracy

How GroundTruth helps you improve the accuracy of your data labeling

`2 core aspects —`<br>

a. **Innovative UX Techniques**<br>
     Directly built into these templates that you can use for common
     labeling tasks<br>

b. **Built-in Algorithms**<br>
     Help improve the accuracy of labeling by taking in the inputs of
     workers and outputting a high fidelity label<br><br>


## Automatic Labeling

How GroundTruth improves the efficiency of data labeling<br>

GroundTruth has an innovative feature called **"Automatic Labeling"**,
which automatically labels a subset of your dataset<br>
A portion of your dataset will still be labeled by humans<br><br>

**[How it actually works]**<br>

It uses an innovative ML technique called **Active Learning**, that
helps us understand which data is well understood and can be potentially
automatically labeled, & which data is not well understood amd may need to
be looked at humans for labeling<br>

GroundTruth can actually look at your training dataset and identify which data is
not well understood and thus needs to be sent to humans.<br>

Under the hood, GroundTruth is actually training an ML model in your
SageMaker account<br>
This whole process is iterative and GroundTruth breaks up your data into
batches and repeats this cycle until all your data is labeled<br><br>

## What does this mean for you?

1. Lowers your total costs of data labeling by up to 70%

2. Enables you to securely manage your training datasets

3. Increases the accuracy of your training datasets<br><br>

## Demo

`3 aspects of a data labeling job`<br>

I. First of all, you need to **provide the input dataset**<br>

You provide us an input manifest, a JSON document containing
Amazon S3 links for each of the separate data points<br>
You also provide us where we'll output the labeled data<br><br>

II. Then you need to tell **what needs to be performed on those datasets**
    (the data labeling task)<br>

* `Image classification` — Categorize images into specific classes
* `Bounding box` — Draw bounding boxes around specified objects in your images
* `Text classification` — Categorize text into specific classes
* `Custom` — Build a custom annotation tool for your specific use case<br><br>

III. Finally, you need to **configure a workforce**<br>

Using Mechanical Turk means you have to check 2 boxes, which are<br>

a. Ensuring the dataset does not contain adult content<br>
b. Understanding that the dataset will be viewed by the Amazon Mechanical
     Turk public workforce and acknowledging that the dataset does not
     contain personally identifiable information (PII)<br><br>


Once the data labeling job is completed, it will take the outputted
labels, augment that initial manifest that you provided, and drop that
into your S3 bucket.<br>

The output manifest will contain the input manifest with the following —<br>
+ the associated label<br>
+ associated metadata for that label<br>
+ confidence score<br>
+ what the actual label category will look like<br>
+ info about whether it was `human-annotated` or `auto-annotated`<br><br>

***

# 3. Introduction to Amazon SageMaker Neo
<br>

## In this course
1. Customer Challenges
2. Service Overview
3. Use cases

## Customer Challenges

Nowadays, ML is universally used by customers to make informed
business decisions.<br>

Developers who want to use ML encounter hurdles at every step<br><br>

## Challenge — Operationalization

1. **(Framework)**<br>
    Choose a framework that's best suited for the task at hand<br>

2. **(Models)**<br>
    Build the models using the chosen framework<br>

3. **(Train Models to Make Predictions)**<br>
    Train a model using sample data to make accurate predictions on
    bigger datasets<br>

4. **(Integrate)**<br>
    Integrate the model with the application<br>

5. **(Deploy)**<br>
    Deploy the application, the model, and the framework on a platform<br><br>

Amazon SageMaker Neo makes it easy to build ML models and get them
ready for training by providing everything you need to quickly connect
to your training data<br>

It also helps you to select and optimize the best algorithm and framework
for your application<br>

Build + Train + Deploy ==> ML Models<br>

However, developers still face hundreds of hurdles when they try to
deploy a model with optimal performance on multiple platforms<br><br>

## How to Deploy a Model With..

Let's review a few details about the realities of model deployment<br>

To begin with, developers need to have the specific version of the
framework used for building the models installed on all of their
chosen platforms<br>

However, platform vendors often limit support to just one or two
frameworks because of the cost and complexity of optimizing and
installing those frameworks<br>

Not to mention the size of the frameworks themselves, which limit the
type of platform on which it can be installed<br><br>


To work around these limitations, developers either restrict the model
deployment to fewer platforms or worse, use a framework that can run on
more platforms even if it's not the best-suited framework for the task
at hand.<br>

In this situation, developers must manually tune the performance of their
models on each platform, which requires rare skills and consumes a
considerable amount of time<br>

As a result, developers often have to settle for sub-optimal performance
in their models, which in turn underutilizes the platform resources
and increases the overall cost of the model's deployment<br><br>


Ultimately, `these hurdles restrict the diffusion of innovations in ML`<br><br>


## Problem — Many to Many

Developers find themselves in a situation requiring their solutions
to run on multiple platforms using multiple frameworks<br>

A ML framework provides functions and operations that developers use
as building blocks for their models.<br>
These functions and operations differ from framework to framework.<br>

Also, each framework has its own mechanism to save the models and often
has unique file formats<br><br>

At runtime, the framework reads the model to generate the code and runs
that code on each unique platform<br>

This means, that the model can only run on the framework on the platform
in which it was built, & only on the hardware configuration on which the
framework is supported.<br>

Do you see the problem here?<br>

It's painful to have all of the platforms support all of the models from
all of the frameworks.<br>
This solution is super messy<br><br>


Amazon SageMaker Neo is designed to solve this messy problem<br>

It `frees a model from the framework in which it was developed and allows
it to make the best use of the hardware on which it was deployed`<br><br>

## Service Overview

**Amazon SageMaker Neo** is a new `SageMaker capability that helps developers
take models trained in any framework and port them on to any platform
while increasing their performance and reducing their footprints`<br>

Neo provides model portability by converting models written in **Apache
MXNet, PyTorch, TensorFlow, and XGBoost** from framework specific formats
into a portable code<br>

During conversion, Neo automatically optimizes the model to `run
up to 2x faster` without a noticeable loss in accuracy<br>

After conversion, the model no longer needs the framework, which
`reduces its runtime footprint on the development platform by a 100 times.`<br>

Customers can access Neo from the SageMaker console, and with just a few clicks,
generate a model optimized for their specific platform<br><br>

## Neo Components

Neo contains 2 components — **a compiler, and a runtime library**<br>

First, the Neo compiler reads models saved in various formats.<br>
Those formats use framework-specific functions and operations<br>

Neo converts those specific functions and operations into
non-specific functions and operations often called
**Framework Agnostic Representations**<br>

Next, Neo will perform a series of optimizations<br><br>

## How Does Neo Work?

Here's a high-level explanation of how Neo could work.<br>

First, the user submits the compilation job requests to Amazon SageMaker,
and the job status is changed to `Starting`.<br>

After a few moments, the job status will change to `In-Progress`<br>
* This is when the process of compiling the model really begins,
  it can take a while<br>

* Now during this time, the user has the ability to stop the job
  if they need to for any reason<br>

When the user stops the job, the status will be changed to `Stopping`;<br>
Amazon ECS will then begin the process of stopping the compilation.<br>

Once that is completed, the job status will then be changed to `Stopped`<br><br>


In most cases, the user is not going to want to stop the job right in the middle<br>

Amazon ECS will then be allowed to complete the compilation.<br>
Once this is completed, the job status will then be changed to `Complete`<br><br>


## Amazon SageMaker Neo Benefits

Earlier, I mentioned the burdens developers carry when it comes to developing
Agnostic ML models<br>

The prior lack of simple-to-use profilers and compilers in ML had forced
developers into a manual trial and error process that is unreliable and
quite frankly, unproductive<br>

Enter Amazon SageMaker Neo.<br>

Neo provides developers with a simple, easy-to-use tool to perform
the conversion of framework specific models and port them to any platform<br>

Neo automatically optimizes the model to run up to 2x faster without a
noticeable loss in accuracy<br><br>


To optimize the model, Neo detects and exploits patterns in the input data,
model computation and platform hardware.<br>

After conversion, the model can run without needing any framework, which
dramatically reduces the model's runtime footprint.<br>

This allows the model to run in resource constrained edge devices, and
performance-critical Cloud services.<br><br>

## Use Cases

Neo enables and accelerates ML models both in the Cloud and at the Edge.<br>

For mobile phones and IoT devices, Neo can help developers better deploy
image classification, object detection, and anomaly detection applications
by relaxing multiple constraints on those devices.<br>

In the Cloud, Neo compiles models to handle data stored in Amazon S3 buckets,
or coming in as data streams<br><br>


Neo also enables new use cases, such as the integration of ML with databases<br>

Developers will be able to take ML models produced by Neo and run them in
databases in the form of user-defined functions<br>

These models will enable complex predictive queries that are currently
impossible or unwieldy to express in ANSI standard SQL language<br><br>

## Key Takeaways

1. **Popular DL and decision tree models**<br>

   Amazon SageMaker Neo supports the most popular DL models that power
   computer vision applications and the most popular decision tree
   models currently used in Amazon SageMaker<br>

2. **Apache MXNet, TensorFlow, PyTorch, XGBoost**<br>

   Neo optimizes the performance of `AlexNet, ResNet, VGG, Inception,
   MobileNet, SqueezeNet, and DenseNet models` trained in MXNet, TensorFlow,
   and even PyTorch<br>

   It also optimizes the performance of classification in `Random Cut Forest
   models` trained in XGBoost<br>

3. **Various Amazon EC2 instances and edge devices**<br>

   Neo supports many SageMaker instance types as well<br>
   It supports **AWS DeepLens, Raspberry Pi, Jetson TX1 or TX2 devices,
   Amazon Greengrass** devices based on Intel processors, as well as
   **NVIDIA Maxwell and Pascal GPUs**<br>

4. **Up to 2x performance speedup and 100x memory footprint reduction
   at no additional charge**<br>

   Developers can train models elsewhere and use Neo to optimize them
   for SageMaker ML instances or Greengrass supported devices<br>

   Neo provides model inference modules which can `run up to 2x faster`, while
   occupying a dramatically reduced memory footprint versus previous
   solutions<br>

   You can use the Neo API to generate an optimized model for your desired
   platform, & you can even deploy and run that optimized model on your
   desired platforms at no additional charge<br><br>

***

# 4. Automated Model Tuning using Amazon SageMaker

**Amazon SageMaker** at a high level, is a `ML platform that we've designed to
make it very easy to build, train and deploy your ML models` to get them from
idea into production as quickly and easily as possible.<br>

These 3 components are interconnected, but independent; so you can use one or
more of them to suit your needs<br><br>


For the **Build** component, we have the ability to very quickly and easily
set up an instance that's running a Jupyter notebook server
(an interactive environment designed for data scientists to explore data,
create Markdown and documentation, and interactive visualizations of data).<br><br>


**Training** is a distributed managed environment, when you create a training
job we spin up a cluster of training instances for you;<br>
we load a Docker container that has an algorithms within it,<br>
we bring in data from S3, we train that algorithm,<br>
we output the artifacts back to S3 and then tear down the cluster
without you having to think about any of that.<br>

We manage that process on your behalf<br><br>


**Deploy** — Once you've trained your model, you can easily deploy it to a
real-time production endpoint, then invoke that endpoint to get real-time
predictions from that ML model<br><br>

On top of these core components, we have additional layers<br>
We have custom provided SageMaker algorithms and these have been designed
from the ground up with advancements in science and engineering<br>

The methodology is slightly different.<br>
It's designed to be more scalable, more efficient, as well as engineering
advancements to use things like GPU acceleration and train in a distributed
setting<br><br>

We also have pre-built DL frameworks for **TensorFlow, MXNet, PyTorch and Chainer**<br>

These frameworks allow you to very quickly and easily write the code that you would
naturally for those DL frameworks, and them deploy them to SageMaker without having
to think about managing the container and knowing that the container that we've set
up for you, pre-built, will allow you to train in a distributed setting & take
advantage of other nice functionalities within each framework<br>

Finally, you have the ability to `bring your own Docker container`<br>
So you can code up your algorithm, package it up in a Docker container &
still take advantage of SageMaker's managed Training and Hosting environments<br><br>

SageMaker's automated model tuning is a service that wraps up training jobs
and works with the custom algorithms, the pre-built DL frameworks, or bring your own
in order to help you find the best hyperparameters and improve the performance of your
ML model<br><br>

## Hyperparameters

**Hyperparameters** `help you tune your ML model in order to get the best performance`<br>

So if you're building a **neural network**, you may want to tune the —<br>

* `learning rate` (the size of updates you make to weights in each iteration of your network),
* `no. of layers` (how deep or shallow your NN is),
* `regularization` (will penalize large weights) and
* `dropout` (will actually drop nodes out of your network to prevent overfitting)<br>

These hyperparameters are important to make sure that you get the best predictive
performance out of your NN<br><br>


Hyperparameters are also used in other ML algorithms like **Trees**<br>

So if you're fitting a decision tree, random forest, or a gradient boosted ensemble
of trees, the<br>
* `number of trees` is important
* `depth` (how deep each tree should be; smaller no. of very deep trees OR large number of very shallow trees),
* `boosting step size` (from round to round of boosting, how much you change)<br>

These things are all very impactful from a supervised learning perspective
in how your model performs<br><br>


Even with **unsupervised ML techniques like clustering**, there still are hyperparameters <br>
* `no. of clusters` you want in total
* `Initialization` (how you want to initialize the seeds in your clustering process)
* `Pre-processing` (whether you want to pre-transform or preprocess the data prior to using
                  the clustering algorithm)<br><br>


## Hyperparameter space (H-space)

1. **Large influence on performance**
2. **Grows exponentially**
3. **Non-linear / Interactions**
4. **Expensive evaluations**<br>

When we think about the H-space and what we need to do to change them
in order to get a better fit, it's important to realize that there's a very large influence of
hyperparameters on overall model performance<br>

Varying hyperparameters can make a very big difference in performance which can obviously
make a big difference in your bottom line from the predictions that you're generating from those
models<br><br>

The H-space also grows exponentially<br>
If we wanted to train > 2 hyperparameters, we'd have to evaluate more and more points in order to
create a good plot ==> gets very hard as we want to tune a large no. of H to do that efficiently<br><br>


There's also non-linearities and interactions b/w the hyperparameters<br>
e.g. when you change one parameter without changing the other, you get one result,
     when you change them together, you get a different result<br>

**Non-linearities** — you `can't just continue to increase one H value and always
                   expect to get better performance`<br>

You'll increase it up to a certain point, at which point in time you'll have
either diminishing returns or even have worse results<br><br>

Each point along the chart would be an expensive evaluation<br>
We'll have to retrain the model for those H value combinations;
so depending on our model complexity and data size, it can be very
expensive to calculate different types of H combinations<br><br>

## Tuning

Several common ways of tuning them —<br>

1. **Manual**<br>
   * Defaults, guess, and check
   * Experience, intuition, and heuristics

2. **Brute force**<br>
   * Grid
   * Random
   * Sobol

3. **Meta model approach**<br><br>


**(Manual)**<br>
  You start with the default value of hyperparameters, you `make a couple of guesses
  and check` and eventually converge on a set of H values that you're comfortable
  with<br>

  The second would be to use `data scientists' experience or intuition` —<br>
  If they've seen a problem like this before, they may be able to pick hyperparameters more
  successfully for future cases<br>

  There also may be `heuristics`, where you can tune one H value first, then subsequently train
  a second and then a third<br>
  These tend to require someone with an advanced skillset in ML to do this process through<br><br>

**(Brute force)**<br>
  The second method. There are 3 common sub-classes — Grid, Random, Sobol<br>

  * With **Grid**, we `try a specific subset of H values for each H`<br>
    e.g. 3 values of one H and 3 values of another ==> combinatorial combination of 9 possible H pairs<br>
    We try each of these in a brute force sense<br>

  * **Random**, we just `randomly pick values for each of the 2 hyperparameters.`<br>
    Although this sounds naive, it's actually quite successful<br>

  * **Sobol**, which tries and blends the two methods above, where you're `trying to fill
    up the space like a grid but you add some randomness in`<br>

    The reason you want to add that randomness in and why random H search can be so
    effective is that very often, we have one H that has a much larger impact on the objective
    or the training loss or accuracy of our model..<br>

    Because of that, Random will try a much larger number of distinct values in that parameter
    and will explore that space better than Grid, where you've predefined a very specific subset
    of values which may miss out on the best performance<br><br>

**(Meta model)**<br>

Tries to `build another ML model on top of your first one` to predict which hyperparameters are going to
yield the best potential accuracy or objective metric results<br>

That's the method that SageMaker takes.<br><br>

## SageMaker's method

* `Gaussian process regression` models objective metric (accuracy, training loss, etc.)
  as a function of hyperparameters<br>
  
➤ **Assumes smoothness**<br>
   ( for a small change in H value, you won't have a drastically wild change in objective metric )<br>
      
➤ **Low data**<br>
   ( important because it's expensive to continue training these models )<br>
      
➤ **Confidence estimates**<br>
   ( what the value of an objective metric would be at different H values)<br><br>

* `Bayesian optimization` decides where to search next (which H value combination we test next)<br>

➤ **Explore and exploit**<br>
Trying out lots of different H values, and when you find good ones, you're testing
nearby points that maybe are slightly better<br>
      
➤ **Gradient free**<br>
You don't have to understand exactly how our objective metric relates to our hyperparameters,
because usually that's unknown<br><br>


**[Demo]**<br>

Bayesian optimization portion.. try and blend the uncertainty we have with our prediction of how
good our objective metric might be, in order to define where the next point we test is<br>

We always test the next point where we find our peak expected improvement<br><br>

## SageMaker Integration

How do we integrate SageMaker H tuning into the SageMaker Automated Model Tuning
capability?<br>

It works with SageMaker algorithms, frameworks, and bring your own container<br>

It's very important — it treats your algorithm like a **black box** that it can
optimize over (doesn't have to know exactly what's going on in your algorithm
in order to be effective at tuning its hyperparameters)<br>

**Flat hyperparameters** are provided, these can either have a continuous
numeric value, an integer value or they can have a categorical value
(one of a subset of potential distinct values)<br>

We need your **objective metrics logged** to CloudWatch, this happens naturally
with SageMaker training jobs; anything that you output will be reported in
CloudWatch, so you can easily output the objective metrics that you want to
optimize on and use a regex to scrape that information out<br>

This is what allows us to be so flexible and work in so many different use cases,
these are the only requirements in order to train a model and use SageMaker's
automated model tuning<br><br>

## Demo

https://github.com/awslabs/amazon-sagemaker-examples/tree/master/hyperparameter_tuning

**Automated Model Tuning** — `as it learns which H values are successful, it will start
exploiting them and fitting ever better models`<br>

A.M.T. can provide better results in fewer training jobs.<br><br>

***

# 5. Amazon SageMaker — Object Detection on Images labeled with GroundTruth

## Agenda

1. Introduction to the dataset
2. Introduction to Amazon SageMaker
3. Activity — Download the data and run a labeling job
4. Activity — Download and review labeling results
5. Introduction to object detection and single-shot detection algorithms
6. Activity — Train and deploy the model
7. Introduction to hyperparameter optimization (HPO), automated model tuning
8. Activity — Hyperparameter optimization
9. Examination of HPO results and replacement of production model<br>

`We'll specifically identify the location of honey bees in photos`<br><br>

## Introduction to the dataset

Quick overview of typical tasks in computer vision —<br>
[It's not an exhaustive list but gives a good idea of the domain]

A. **Classification**<br>
   We want to simply determine whether the image does or doesn't contain a
   certain single object<br><br>

B. **Classification + Localization**<br>
   Next level of complexity is actually identifying where in the image the object is,
   for example, with a simple bounding box<br><br>

C. **Object Detection**<br>
   More generally, instead of a single object we'd like to identify as many
   different objects in a scene as possible, giving rise to the object detection
   problem<br><br>

D. **Instance Segmentation**<br>
   In some applications, it might be necessary to be more precise than the
   rectangular bounding box, and this leads to an instance segmentation task
   where each pixel is classified as either belonging or not belonging to a
   particular object<br><br>

iNaturalist.org crowdsourcing project that comes with a website and a handy
smartphone app allowing people to upload photos of their plants, animals, etc.
tagged with location, date, etc.<br>

The purpose is to record the sighting but also to help identify the species
based on established biological taxonomy relying on community experts.<br><br>

### iNaturalist.org — How it works

1. Record your observations
2. Share with fellow naturalists
3. Discuss your findings<br>

iNaturalist also exposes a handy expert functionality where you can download
the observation details based on selection criteria such as species, the
geography, etc.<br>

When people upload their photos, they choose what license they prefer to share
them under.<br>
We'll use 500 photos of honeybees under Creative Commons public domain (CC0) license<br><br>


When it comes to downloading, we can choose all the desired attributes of an
observation<br>
e.g. to know under which license the images are provided, choose the `license`
attribute<br><br>

## Introduction to Amazon SageMaker — Build, train, and deploy ML

Amazon SageMaker is designed to eliminate the heavy lifting from all parts of
a typical process of building, training, and deploying ML models.

1. **Pre-built notebooks for common problems**<br>

First off, it comes with a Jupyter notebook server that lets you manage notebooks.<br>
There are dozens of pre-built notebooks to help you start solving many common ML
problems<br>

2. **Built-in high performance algorithms**<br>

You can customize these to your task, taking advantage of the many built-in ML
algorithms, or create your own ML recipe using one of many popular ML frameworks
such as MXNet, TensorFlow or PyTorch<br>

3. **One-click Training**<br>

You can kick off training of a developed model using as much computational power
as you need by choosing the right size cluster of GPU or CPU-based machines
available in the AWS Cloud<br>

4. **Optimization**<br>

Even the most sophisticated algorithms still have many parameters one must specify
in order to start training<br>
Finding the right values for such parameters feels often more like an art than science<br>

This is why SageMaker provides `hyperparameter optimization (HPO)`, a.k.a. automated model
tuning<br>

5. **One-click Deployment**<br>

Finally, once a satisfactory model is trained, SageMaker makes it easy to create a
scalable production inference endpoint by taking care of model deployment<br>

6. **Fully managed with auto-scaling**<br>

Autoscaling is supported out of the box<br><br>

SageMaker is a powerful tool and that's the reason many companies choose to adopt it
including Intuit, F1, Tinder, Siemens, Thomson Reuters, GE Healthcare, etc.<br><br>

### GroundTruth is an extension of the Amazon SageMaker platform

> Label ML training data easily and accurately

I. **Ground Truth**<br>

* Set up and manage labeling jobs for highly accurate training datasets using active
learning and human labeling<br>

II. **Notebook**<br>

* Availability of AWS and SageMaker SDKs & sample notebooks to create training jobs
and deploy models<br>

III. **Training**<br>

* Train and tune models at any scale
* Leverage high performance AWS algorithms or bring your own<br>

IV. **Inference**<br>

* Create models from training jobs or import external models for hosting to run inferences
on new data<br><br>

In addition to the 3 main functional blocks in SageMaker of notebooks, training, and inference,
many additional features are built into the servers.<br>

We'll use SageMaker GroundTruth to label the raw image dataset, identifying the exact location
of honeybees in each.<br><br>


For many practical problems, we may have collected the data needed for training, yet this data is
missing the correct target attribute which is required for ML<br>

In our case, such a target attribute is the bounding box or generally, many bounding boxes
around each honeybee in an image<br><br>


Labeling a dataset requires human input, after all we're training a ML model to mimic
human decisions<br>

Not only is there the labor cost involved in labeling each image, but we want to involve
many human labelers in parallel so that a large dataset can be processed as fast as possible<br>

SageMaker GroundTruth makes all of this possible and that is why for many real life problems,
GroundTruth lets you turn a potential intractable problem due to cost and needed orchestration,
into a solvable one, enabling more and more ML applications<br><br>

### Use pre-built labeling workflows or set up your own

GroundTruth supports these labeling tasks straight out of the box —<br>

* **Bounding Boxes** (Object detection)
* **Image Classification**
* **Text Classification**
* **Semantic Segmentation**<br>

This includes the labeling user interfaces required for such tasks<br>

SageMaker also lets you build a custom labeling workflow<br><br>

### Easily set your own custom labeling workflow

1. Pre-processing AWS Lambda function
2. Labeling UI template
3. Post-processing Lambda function

Multiple options — Use a provided UI template (15+ to choose from), use
<i>Crowd HTML Elements</i> to construct your own UI, or bring your own
custom HTML or JavaScript<br>

For e.g. you may simultaneously need to label an image and a related piece of text<br>
An accustomed labeling template would be required in such a case<br>

The UI can be constructed using <i>Crowd HTML Elements</i>, a tag library
defined by **Amazon Mechanical Turk**<br>

Moreover, you can provide logic in the form of Lambda functions that do
pre- and post-processing of labeling data<br>

E.g. you may want to use a custom algorithm to score the results of the labeling task<br><br>


If you need the help of many humans to label the dataset, where do you get these helpers?<br><br>

### Select your pool of labelers

SageMaker GroundTruth `supports 3 different types of labelers` —<br>

I. **Public**<br>

An on-demand 24x7 workforce of over 500,000 independent Contractors
worldwide, powered by Amazon Mechanical Turk<br>

II. **Private**<br>

A team of workers that you have sourced yourself, including your own
employees or contractors for handling data that needs to stay within
your organization or sensitive data that can't be exposed publicly<br>

III. **Vendors**<br>

A curated list of third party vendors that specialize in providing data
labeling services, available via the AWS Marketplace<br><br>

### Use automated data labeling to save cost and time

Another important value out of GroundTruth is the `ability to learn while the
human labeling is in progress` — **Active Learning**.<br>

If the active learning model built in real time is confident that it can
automatically label image it will do so, and if it's not, it will still send
it to a human labeler<br>

For large datasets, this feature can dramatically reduce the overall labeling cost.<br><br>

## Demo — Downloading the Jupyter notebook

http://aws-tc-largeobjects.s3-us-west-2.amazonaws.com/DIG-TF-200-MLBEES-10-EN/demo.ipynb

Jupyter notebooks are also known as IPython notebooks ==> .ipynb extension<br>

dataset.zip contains training files, 10 files for testing, and the
`output.manifest` file — contains the results of GroundTruth labeling job for all
500 images of bees present in the dataset<br><br>

## Object detection in computer vision
<br>

### Object detection with deep learning

https://gluon-cv.mxnet.io/build/examples_datasets/detection_custom.html

**Bounding box**<br>
* Generated by model
* Represented by (h, w, x, y)<br><br>

Given a photo containing one or more objects, our goal is to find the tight
bounding box around every object that the model can identify, together with
the corresponding class label<br>

Each bounding box is described by the position of the top-left corner: x and y
coordinates, as well as height and width<br>
These could be expressed in absolute numbers of pixels or as percentages relative
to the overall image width and height<br><br>

### Object detection — Choosing patches

1. Multiple objects to be recognized in a single image will require multiple
   outputs from the neural network

2. An intuition<br>
* Choose "in some way" areas in an image and extract patches
* Pass the patches through a CNN to get the classification<br><br>

This means that for a neural network to detect many different objects, it
must be able to produce a separate output for each object, incorporating the
bounding box location and class label, as well as the corresponding
classification confidence<br>

Intuitively, we can imagine the process of object detection as occurring in 2
separate steps<br>

* One step is proposing interesting regions where the object might be, and
  another is classifying the objects in the region by generating bounding boxes<br><br>

### Selective search

http://www.huppelen.nl/publications/selectiveSearchDraft.pdf

1. By generating initial sub-segmentation, you can generate many candidate regions
2. Use greedy algorithms to recursively combine similar regions into larger ones
3. Use the generated regions to produce the final candidate region proposals<br>

One approach is to first apply pixel-level sub-segmentation and then apply a greedy
algorithm for merging together similar regions.<br>

Once larger sub-regions have been obtained, candidate regions for where the object might
reside are proposed<br><br>

### Single-shot detectors

https://arxiv.org/pdf/1506.02640v5.pdf

* Instead of having two networks — the Region Proposal Network and the Classifier Network,
  **single-shot architectures** have `bounding boxes and confidences for multiple categories` and are
  `predicted directly with a single network`<br>

Turns out, such a two-step process can be optimized to be completed in a single forward pass, namely
via the single-shot detectors or SSDs<br>

All of this happens within a single neural network<br><br>

### Single Shot Multibox Detector (SSD)

https://arxiv.org/pdf/1512.02325.pdf

Reached new records in terms of performance and precision for object detection
tasks (`74% mAP at 59 FPS`) on standard datasets<br>

* **Single Shot** — Classification and localization done in a single forward pass

* **Multibox** — Technique for bounding box regression

* **Detector** — An object detector that also classifies the detected objects<br><br>

[slide shows a SSD network based on VGG16 CNN]<br>

Different SSD topologies also exist e.g. based on ResNet-50 network<br>

Such SSD systems have shown superior performance, both in terms of
accuracy and inference speed<br><br>


Speaking of accuracy, how do we measure if the model produced good
results?<br>

Unlike traditional ML classifiers, not only the model needs to produce
an accurate classification, but also generate an accurate tight bounding
box around the object<br>

The bounding box could be off by a few pixels, or could be omitting part of
the object or being a completely wrong place<br>

How do we then compare accuracy of different algorithms?<br><br>

### Intersection over Union (IoU)

One idea is to `take the predicted bounding box and the true bounding box
and measure the degree to which these boxes overlap`<br>

For e.g., we could compute the ratio of two areas — intersection area
divided by union area or **IoU**<br>

For perfectly matching bounding boxes, this ratio will be one, whereas
for non-intersecting bounding boxes it will be zero<br>

The metric that is actually used is called **Mean Average Precision (MAP)**,
it's also based on the IoU concept<br><br>

## Demo

Input mode —
1. `File` — downloads the entire dataset onto the box that's performing training
2. `Pipe` — delivers the data in just-in-time streaming fashion<br>

If you have lots and lots of images, it's more efficient to have a larger
batch size<br><br>

## Automated model tuning with Amazon SageMaker

We need to supply many different hyperparameters for our object
detection built-in algorithm<br>

If you don't have lots of experience for a particular algorithm
and ML problem, it will likely be very difficult for you to
decide what set of parameters is best<br>

The reality is that even the most experienced and trained
practitioners often need to explore the space of hyperparameters
to see which ones affect the model performance the most<br>

That's why we've added a way to automatically find the best set of
hyperparameters, a feature called **automated model tuning**<br>

In the industry, it's also known as **Hyperparameter Optimization or HPO**<br><br>


### Hyperparameters

Hyperparameters exist in most ML algorithms<br>

In the case of **DL**, which is based on neural networks, typical H
consist of<br>
* `the learning rate`,
* `the no. of layers`,
* `regularization and dropout` (as means of dealing with overfitting)<br>

For more traditional **decision tree-based algorithms**, such as popular
XGBoost, these could be<br>
* `the no. of trees in the ensemble`,
* `the maximum depth of the tree`, and
* `boosting step size`<br>

For **clustering**, <br>
* `No. of clusters`
* `Initialization`
* `Pre-processing`<br><br>

Each H has a corresponding set of possible values and altogether
the hyperparameters for a particular algorithm form a H-space
that we need to explore to find the best point in that space<br><br>


### Hyperparameter space

* **Large influence on performance**<br>

The effect on performance can be quite dramatic<br>

[ slide contains graph of different values of `embedding_size` and `hidden_size`
and the validation_f1 score as a result of their combinations ]<br>

Validation F1 score characterizes the accuracy of the resulting model<br>

`embedding_size` in this case could be the dimensionality of the word embedding
in a NLP problem, while the `hidden_size` could be the size of the hidden layer in a
neural network<br>

The resulting difference between a validation F1 score of 87 versus 95 could be the
difference between an acceptable or an unacceptable ML model<br><br>

* **Grows exponentially**<br>

The H-space here contains just 2 hyperparameters but usually we have many more<br>

It quickly becomes difficult to understand and explore the space by hand which is
what researchers often had to do before automated HPO<br><br>

* **Non-linear / interactions**<br>

Additionally, hyperparameters are typically not independent of each other in terms of their
combined effect on the model — a change in one would likely affect how another influences
the model<br><br>

* **Expensive evaluations**<br>

Often, the only way for you to know how the model is going to react to something different
is to make changes to hyperparameters and train a new model<br>

Exploring the space exhaustively in this way is usually going to be quite costly<br>

So what are the typical approaches to model tuning?<br><br>

### Tuning

http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf

1. **Manual**<br>
   * Defaults, guess, and check
   * Experience, intuition, and heuristics<br>

This approach at best is inconsistent and relies on a lot of previous experience<br>

2. **Brute force**<br>
   * Grid
   * Random
   * Sobol

Early attempts at automation relied on various brute force approaches.<br>

One idea is to simply divide the ranges of the hyperparameters into same size
intervals, and then effectively do a **grid search**<br>
Problem — usually some parameters are highly important to improving the performance
and others are not<br>

**Random** — Choose the positions randomly<br>

**Sobol** algorithm used to generate quasi-random numbers<br>
These are better than pseudorandom numbers in that they sample the space more evenly,
avoiding clustering<br><br>

### The Amazon SageMaker method

`Gaussian process regression` models objective metric as a function of hyperparameters<br>
* Assumes smoothness
* Low data
* Confidence estimates<br>

`Bayesian optimization` decides where to search next<br>
* Explore and exploit
* Gradient free<br>

While also offering random search, SageMaker provides a smarter approach based on
Bayesian optimization, based on Gaussian process regression which provides estimated
bands for objective parameter in the yet to be explored areas, and the algorithm chooses
the most promising area to explore next<br><br>

***

# 6. Build a text classification model with Glue and SageMaker

SageMaker notebooks are good for learning, and use with small datasets but with real use cases and
real-life datasets, the notebooks can't scale up to the challenge even with accelerated computing
instances<br>

This is where AWS Glue becomes really relevant — you can work with enormous amounts of data, scale
and process that data in order to be consumed by SageMaker or any algorithm that you want to bring<br><br>

## Why should you care about Text Classification?

We produce a ton of text e.g. Twitter posts, news<br>
We may want to leverage that text in order to improve our business<br>

* Triaging trouble tickets<br><br>

## The ML process

4 different stages, each starts with the business problem<br>

Once you have defined the problem, and some requirements about accuracy, then you
need to have data (**Data Collection**)<br>

Integrate that data (**Data Integration**)<br>

(**Data Preparation & Cleaning**)<br>
(**Data visualization and analysis**) ==> Feature engineering ==> Model training & parameter tuning ==> Model evaluation<br>
(**Model deployment**)<br>
(**Monitoring & debugging**)<br><br>


## The Process in Practice

### 4 phases of ML

1. **Discovery**

> Framing our business problem

<code>**Goal** — Automate Trouble Ticket Triaging</code><br>

* Streamline ticket creation for customers (reduce metadata used)
* Reduce ticket solving time
* Minimize bouncing tickets<br>

**Dataset — Amazon Reviews dataset**<br>

* Author
* Rating
* Header
* Body
* Usefulness (how many people found it useful)<br><br>

https://s3.amazonaws.com/amazon-reviews-pds/readme.html

* 20 years of product reviews from Amazon.com customers with accompanying metadata
* Over 160 million reviews
* ~80 GB of raw data (~51 GB parquet compressed)
* Available in a publicly accessible S3 bucket in N. Virginia
  TSV — s3://amazon-reviews-pds/tsv/
  Parquet — s3://amazon-reviews-pds/parquet/
* Partitioned by `product_category`<br>

**Similar problem, different business**<br>


> Predict a product category based on the review text


**How good should the model be?**<br>

A **confusion matrix** is a `measure of how good a classification model works`<br>
y-axis — True labels of data<br>
x-axis — Predicted label<br><br>


2. **Integration**<br>

> Visualizing and analyzing the data

To do that, we can use **AWS Glue Crawler**<br>

A **crawler** is a `process that will traverse our data and try to extract a schema
out of it` ==> use the extracted table with AWS services like **Athena**<br><br>

(**Query**)<br>
You can use QuickSight to make visualizations on data that is being created by Athena<br>

Dataset is unbalanced, reviews on books are the highest than any other product categories<br>
This would lead to a biased model (predictions would lean towards 'books' because the algorithm
has been trained on a lot of books)<br>

We'd like to go from this to a balanced dataset containing roughly equal number of product reviews
per category<br><br>


> ETL — T is for Transformation

We'll use Glue for that

`Glue Development Endpoints + SageMaker Notebook Instances`<br>
Deploy an endpoint ==> access it from notebook<br><br>


*Balancing the Dataset*

a. **Different strategies**<br>
   * Duplicate records
   * Remove records (we'll go ahead with this)
   * Sophisticated stuff<br>

b. **How many and which to remove?**<br>
   * Equalize with the category with lowest count
   * Remove randomly<br><br>

**TODOs**

* <code>Find the category with the lowest count and calculate a sampling factor (N<sub>i</sub>) for each category
  <i>i</i></code><br>

  ```
  # Read data from the source with a Glue DynamicFrame
  datasource = glueContext.create_dynamic_frame.from_catalog(
    database = database,
    table_name = table
  )

  # Convert the DynamicFrame to a Spark DataFrame
  df = datasource.toDF()

  # Number of reviews per category
  per_category_count = df.groupBy('product_category').count().collect()

  # Find the category with least reviews
  counts = [x['count'] for x in per_category_count]
  min_count = float(min(counts))

  # Calculate the factor to apply to each category and put them in a tuple
  factors = map(lambda x: (x['product_category'], min_count/float(x['count'])),
       per_category_count)
  ```
  <br>

* <code>Take a sample of N<sub>i</sub> reviews for each category <i>i</i></code><br>

  ```
  # Pick the corresponding sample of each category and put them on a list
  samples = []

  for category, n in factor:
    sample = glueContext.create_dynamic_frame.from_catalog(
      database = database,
      table_name = table,
      push_down_predicate = "product_category == '{}'".format(category) # leverages data partitions
    )

    sample = sample.toDF().sample(
      withReplacement = False,
      fraction = n,
      seed = 42
    )

    samples.append(sample)
  ```
<br>

* **Write samples to S3**<br>

```
# Build a Spark DataFrame that is the union of all the samples
balanced_df = samples[0]

for sample in samples[1:]:
  balanced_df = balanced_df.union(sample)

# Convert the DataFrame to a Glue DynamicFrame
balanced = DynamicFrame.fromDF(balanced_df, glueContext, "balanced")

# Write the data on the target bucket using parquet format
sampled_data_sink = glueContext.write_dynamic_frame.from_options(
  frame = balanced,
  connection_type = 's3',
  connection_options = {"path": target, "partitionKeys": ["product_category"]},
  format = "parquet"
)
```
<br>

3. **Training**<br>

> Choosing an algorithm and preparing the data for it

**SageMaker BlazingText**<br>

Two modes —<br>

1. **Unsupervised**<br>
  * Highly optimized implementation of **Word2vec**
  * Used for converting words to vectors (a.k.a **word embeddings)<br>

**word embedding** = `representation of a word in the form of a vector`<br><br>

2. **Supervised**<br>
  * Extends the *fastText* text classifier
  * Used for multi class / label text classification<br><br>

We'll use Supervised mode<br>

For BlazingText to work, we need to provide data to it in a specific way<br><br>


**Target Format — BlazingText**<br>

* Single preprocessed text file
* Space separated tokens (token = word or punctuation symbol)
* Single sentence per line
* Labels alongside the sentence
* A label is a word prefixed by the string `__label__`
* Can use training and validation channels<br><br>


**TODOs**<br>

1. Select only the used fields
2. Tokenize review body and convert it to a space separated string and
   prepend the string with the product category label
3. Split the dataset into training, validation, and test subsets
4. Write each subset to a single object in S3<br><br>

[Looks like this]

```
__label__4 linux ready for prime time , intel says , despite all the linux hype...
__label__2 bowled by the slower one again , kolkata , november 14 the past caught...
```
<br>


### Preparing the Dataset — AWS Glue Job Script

I. **Select only the used fields**<br>

```
# Read from the data source
datasource = glueContext.create_dynamic_frame.from_catalog(
  database = database,
  table_name = table
)

# Select only the fields that are going to be used
select = SelectFields.apply(
  frame = datasource,
  paths = ["product_category", "review_body"]
)
```
<br>

II. **Tokenize review_body and convert it into a space separated string**<br>

*[Every single role in that data is going to be applied a function]*<br>

```
# Transform the reviews text by applying the tokenize function to each row
# in the DynamicFrame
tokenized = Map.apply(frame = select, f = tokenize, transformation_ctx = "tokenized")

def tokenize(dynamicRecord):
  category = dynamicRecord['product_category'].lower()
  dynamicRecord['product_category'] = '__label__'+dynamicRecord['product_category'].lower()
  dynamicRecord['review'] = transform_review_body(dynamicRecord.get('review_body', ''))
  return dynamicRecord

def transform_review_body(review_body):
  from nltk.tokenize import TweetTokenizer
  tknzr = TweetTokenizer()
  body = tknzr.tokenize(remove_tags(review_body.lower()))
  return(' '.join(body))
```
<br>

III. **Split the dataset into training, validation, and test subsets**<br>

*[60% for training, 20% for validation and testing each]*<br>

```
# Split the sample into train, test, and validation sets.
# A Spark DataFrame is needed for this
df = tokenized.toDF()
train, validation, test = df.randomSplit(weights = [.6, .2, .2], seed = 42)
```
<br>

IV. **Write each subset to a single object in S3**<br>

```
# Repartition the data frames to store each set into a single file and convert to DynamicFrame
train_set = DynamicFrame.fromDF(train.repartition(1), glueContext, "train")
validation_set = DynamicFrame.fromDF(validation.repartition(1), glueContext, "validation")
test_set = DynamicFrame.fromDF(test.repartition(1), glueContext, "test")

# Write each set with a data sink
train_datasink = glueContext.write_dynamic_frame.from_options(
  frame = train_set,
  connection_type = "s3",
  connection_options = {"path": "{}/train".format(target)},
  format = "csv",
  format_options = {"separator": " ", "writeHeader": False, "quoteChar": "-1"}
)
```
<br>

After this, we'll have the data available and ready for SageMaker to use with BlazingText<br><br>


> Passing the data through the algorithm and creating a model
<br>

**Using the generic Amazon SageMaker Estimator**<br>

* Part of the SageMaker Python SDK
* Needs to be configured with —<br>
  1. Container for BlazingText algorithm
  2. Training and Validation channels
  3. IAM Role
  4. Training instance configuration
  5. Hyperparameters
  6. Output location for the training artifacts<br><br>


4. **Deployment**<br>

`[END OF SECTION]`<br><br>


## Tips to Improve Model Training

1. Shuffle (order by a random value and then repartition)

2. Short Reviews seem to be Useless in many cases — get rid of them<br>

```
SELECT review_body FROM parquet WHERE LENGTH(review_body) < 20 LIMIT 10
```
<br>

## Cost

AWS Glue + Amazon SageMaker Cost<br>

`DPU` = Data Processing Unit<br>

➤ Development endpoint<br>
  * 2 DPUs x 3 hours x $0.44 DPU-hour = $2.64<br><br>
  
➤ Sampling Job<br>
  * 30 DPUs x 6 min (minimum 10) x $0.44 DPU-hour = $2.2<br><br>
  
➤ Preparation Job<br>
  * 20 DPUs x 25 min x $0.44 DPU-hour = $3.67<br><br>
  
➤ SageMaker Notebook instance<br>
  * 1 ml.t2.medium x 3 hours x $0.05 = $0.15<br><br>
  
➤ Hyperparameter Tuning Job<br>
  * 1 ml.c5.4xlarge x $1.075 x 2 hr 58 min total training = $3.19<br><br>

**Total = $11.85**<br><br>


## Glue Jobs Optimization

When you run Glue jobs, you can get metrics which tell you things about
how many executors you need<br>

Executors can be translated into DPUs<br>

Need to find the sweet spot, because it depends on the workload<br><br>

***
