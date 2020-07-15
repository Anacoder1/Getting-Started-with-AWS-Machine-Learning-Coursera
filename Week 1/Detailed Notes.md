☀ [1. What is Artificial Intelligence?](https://github.com/Anacoder1/Getting-Started-with-AWS-Machine-Learning-Coursera/blob/master/Week%201/Detailed%20Notes.md#1-what-is-artificial-intelligence)<br>

☀ [2. What is Machine Learning?](https://github.com/Anacoder1/Getting-Started-with-AWS-Machine-Learning-Coursera/blob/master/Week%201/Detailed%20Notes.md#2-what-is-machine-learning)<br>

☀ [3. What is Deep Learning?](https://github.com/Anacoder1/Getting-Started-with-AWS-Machine-Learning-Coursera/blob/master/Week%201/Detailed%20Notes.md#3-what-is-deep-learning)<br>

☀ [4. Understanding Neural Networks](https://github.com/Anacoder1/Getting-Started-with-AWS-Machine-Learning-Coursera/blob/master/Week%201/Detailed%20Notes.md#4-understanding-neural-networks)<br>

☀ [5. Machine Learning Algorithms Explained](https://github.com/Anacoder1/Getting-Started-with-AWS-Machine-Learning-Coursera/blob/master/Week%201/Detailed%20Notes.md#5-ml-algorithms-explained)<br><br>

# 1. What is Artificial Intelligence?
<br>

## In This Video
* What is AI?
* Why is AI important?
* What are ML and DL?
* How Amazon uses AI in its products?
* AI-supporting services and frameworks available in AWS
* Use Cases

Simply stated, AI is `intelligent behavior by machines`<br>
Any device that can perceive its environment and take actions accordingly has AI

By using AI, a machine can mimic cognitive human functions, like
learning and problem-solving

A common example of using AI is giving machines the ability to scan and interpret
their physical environment, so that they can handle moving around and even up-and-down
the stairs

To make the machines act and react like humans, we need to provide them with
information from the real world<br><br>


### Knowledge Engineering

In order to mimic human intelligence, AI relies on something called **knowledge
engineering**

Knowledge engineering is a key component of AI research<br><br>

Machines with AI are expected to solve problems like humans would

To do that machines need extensive knowledge of the real world

In other words, they need to understand things like the relationships between
objects and situations, the properties of an event, cause and effect, and more

This data is then processed and fed to software programs that in turn analyze the
data and come up with decisions for a particular problem, the way humans do<br><br>


In short, the goal is to transfer human expertise to a software program, that can
take in the same data and come to the same conclusions as humans would

This process of `feeding data to a software program and coming up with human-like
decisions` is also known as the **modeling process**

The model, basically your software algorithm, is consistently refined until its
decisions are close to those a human would come up with

If the decision for a particular problem is inconsistent with what a human
decision would be, then we go back to the model and debug it until we improve it

This is an iterative process.<br><br>


## Why is AI important to you?

* Product enhancements and features
* New user experience; New Markets
* Disruption

Businesses in all domains such as finance, retail, telecom, media, etc. are
innovating with AI

AI presents us with new possibilities and promotes growth in business
All kinds of companies are using AI to innovate

Companies are making significant investment to improve their products based on
user satisfaction, feedback, trends and more, & they are using AI to do it<br><br>

Few examples of `how AI is being used today —`<br>
* Detecting and deterring security threats and fraud
* Resolving users' technology issues through automated call center or chatbot
* Automating repeatable tasks such as payroll, data entry, and auditing
* Anticipating users' actions and providing recommendations
* Monitoring social media comments
* Tailoring advertising content as per search trends


Once you start learning about AI, you start seeing terms like ML and DL

ML and DL are subsets of AI<br>

You can create an AI system with the help of ML and DL algorithms, e.g.
a software program to predict user actions and suggest recommendations
or a system that understands thoughts and sentences spoken by a human,
like Alexa<br><br>


### Machine Learning

ML is often deployed where explicit programming is too rigid or impractical

Unlike regular computer code, ML `uses data to generate statistical code that
will output the right result, based on patterns recognized from previous examples
of input`<br>

ML starts with the data it already has about a situation<br>
It processes data using algorithms to recognize patterns of a behavior and outcomes<br>
It then interprets those patterns to predict future outcomes<br>

These predictions are used to make a decision about the next step for the ML to take<br>
That decision produces results, which are then evaluated and added into the pool of data<br>
The new data would influence the predictions and subsequent decisions made going forward<br>

This is how ML learns over time<br><br>


### What can ML do?

* Make predictions from huge datasets
* Optimize utility functions
* Extract hidden data structures and patterns
* Classify data

This enables a software program to learn and make predictions in the future<br><br>


### Deep Learning

DL takes ML a step further.<br>
Rather than telling the machine what features it needs to look for, DL
`enables the machine to define the features it needs to look for itself,
based on the data it's being provided`<br>

**E.g.** ML requires you to tell the machine how to differentiate between rectangles
and circles<br>
DL on the other hand shows machines several examples of rectangles.<br>
It analyzes those examples and infers common features that define a rectangle<br>

At this point, it can identify on its own whether it's looking at a rectangle<br><br>


In the same way our brains process information using neurons, DL processes information
using similar but artificial processing structures known as **Artificial Neural Networks**<br>

It builds these structures from the data it analyzes and then infers features about
its subject matter based on the data<br>
Then it weighs those features according to a certainty and commonality, & organizes them
into layers of hierarchies and relationships with each other

**E.g.** if the DL machine looks at its reference data on what a rectangle is, it can infer
that rectangles are built from four sides at right angles<br><br>

Unlike ML, the DL machine doesn't have to be told to look for the number or angle of sides;
instead, it recognizes the sides as a common feature of the reference data on its own<br><br>

## How do AI, ML, and DL differ?

In this example, an AI wouldn't necessarily know that it was looking at three people, unless
it has been taught what to look for in order to spot people<br>

This requires a lot of trial and error on the part of the developers creating the algorithm,
and it doesn't involve the machine having to learn anything about what humans look like,
other than what the developers tell it to look for<br>

The machine may be provided with the ability to identify head shapes or skin tones, but without
the ability to learn, the machine could fail simply because of the wide range of diversity in
what humans look like<br>

**For e.g.**, it might not recognize a person because of a beard, which could generate a `false negative`<br><br>


**With ML**, however, you can give the machine `a rough framework for what a person looks like` and the
ability to iteratively process and learn other human appearances through experience<br>

**With DL**, the machine is `provided lots of facial reference data upfront` and unlike traditional
ML or AI, it `isn't always told exactly what features to look for`<br>

It uses its highly advanced data processing capabilities and neural networks to derive the
important features it needs to look for from the data itself

Rather than the developers telling the machine ahead of time how to recognize specificities,
like facial hair, the machine simply looks for the common features that define all of the humans
in this data, and looks for those in the things that it sees

i.e. the machine defines the essential features of its subject, rather than the developer

That's what distinguishes DL from traditional ML<br><br>

## How to Establish an Effective AI Strategy

* Fast computing environments
* Ubiquitous data
* Advanced learning algorithms<br>

You can establish an effective AI strategy in your organization with the help of fast
computing environment, data gathered from various sources such as social media, browsing
trends and more, & advanced learning algorithms<br><br>

### The Flywheel of Data

**More data** means better analytics which results in **better products**<br>
Better products mean **more users** and that in turn generates more data — `Data Flywheel`<br>

You can gather data from a no. of sources like clickstream and user activity, then
you can analyze it using tools like Hadoop, Spark, Amazon Elasticsearch service, etc.

Using the analysis, you can feed the AI and ML algorithms to form pattern recognitions and
generate predictions<br>
Then you can use those predictions to make your products better and drive more users to it<br><br>


By using a combination of programming models, algorithms, data and hardware acceleration with
infrastructure such as GPUs, you can develop a framework that helps with AI-enabled features
like image understanding, speech recognition, NLP, and autonomy

This *combination of programming models and data* is usually what forms the basis of ML and DL
frameworks, the underlying hardware infrastructure supporting the frameworks<br><br>

## AI at Amazon

* Discovery and search
* Fulfillment and logistics
* Enhancing existing products
* Defining new product categories
* Bringing ML to all

AI is used all across Amazon<br>
On Amazon.com, users see recommendations suggested by Amazon's recommendation engine, which
improves their shopping experience<br>

We also use AI to spot trends in the customers' experience so that we can develop new products
and enhance existing ones<br>

In the `fulfillment and logistics department`, robots pick, pile, sort, and move boxes around so
that they can be shipped to customers<br>
Our employees used to have to walk miles each day. By using AI, we save time and free up our
staff to serve more customers faster<br><br>


AWS is making AI tools broadly available so that businesses can innovate and improve their
products<br><br>

## AI on AWS

AWS offers a range of services in AI by leveraging Amazon's internal experience with AI and ML
These services are separated according to 4 layers — `AI [services, platforms, frameworks, infrastructure]`<br>

They organize from the **least complex to the most**, going from top to bottom<br><br>


**AI services** are each `built to handle specific common AI tasks`<br>

These services enable developers to add intelligence to their applications through an API called to
pre-train services rather than developing and training their own DL models


**Amazon Rekognition** makes it easy to `add image analysis to your applications`<br>

* With Rekognition, you can detect specific objects, scenes, and faces like celebrities, &
identify inappropriate content in images<br>

* You can also search and compare faces

* Rekognition's API enables you to quickly add sophisticated DL-based visual search and
 image classification to your applications<br><br>


**Amazon Polly** is a service that `turns text into lifelike speech`, allowing you to create applications
that talk and build entirely new categories of speech-enabled products

* Polly's text-to-speech service uses advanced DL technologies to synthesize speech that sounds like
  human voice<br><br>


**Amazon Lex** is a service for `building conversational interfaces into any application using voice and text`<br>

*  It provides `automatic speech recognition` for converting speech-to-text and NLU to recognize the intent
  of the text

* That lets you build applications with highly engaging user experiences and lifelike
  conversational interactions<br><br>


The **AI Platforms** layer of the stack includes `products and frameworks that are designed to
support custom AI related tasks` such as training a ML model with your own data<br>

For customers who want to fully manage platforms for building models using their own data, we
have **Amazon ML**<br>

*  It's designed for developers and data scientists who want to focus on building models

The Platform `removes the undifferentiated overhead` associated with deploying and managing
infrastructure for training and hosting models<br>

It can analyze your data, provide you with suggested transformations for the data, train your
model and even help you with evaluating your model for accuracy<br><br>


**Amazon EMR** is a `flexible, customizable, and managed big data processing platform`<br>

*  It's a managed solution in that it can handle things like scaling and high
  availability for you

*  EMR doesn't require a deep understanding of how to set up and administer big data
  platforms; you get a preconfigured cluster ready to receive your analytics workload

* It's built for any data science workload, not just AI<br><br>


**Apache Spark** is an open-source, `distributed processing system commonly used for big data workloads`<br>

*  Spark utilizes in-memory caching and optimized execution for fast performance,
  supports general batch processing, streaming analytics, ML, graph databases and
  ad-hoc queries

*  It can be run and managed on Amazon EMR clusters<br><br>


The **AI Frameworks and infrastructure layers** are `for expert ML practitioners` i.e.
for the people who are comfortable building DL models, training them, doing predictions (inference)
and getting the data from models into production applications<br>

The underlying infrastructure consists of `Amazon EC2 P3 instances`, optimized for ML and DL<br>

Amazon EC2 P3 instances provide **powerful NVIDIA GPUs** to accelerate computations, so that customers
can train their models in a fraction of the time required by traditional CPUs<br>

After training, Amazon EC2 C5 compute-optimized and M4 general-purpose instances, in addition to
GPU-based instances, are well suited for running inferences with the training model<br><br>

AWS supports all the major DL frameworks and makes them easy to deploy with our AWS DL Amazon Machine Image,
available for Amazon Linux and Ubuntu, so that you can create managed, automatically scalable clusters of
GPUs for training and inference at any scale<br>

  It comes pre-installed with technologies like **Apache MXNet, TensorFlow, Caffe, Caffe**, &
  other popular ML software such as the **Anaconda package** for data science<br><br>

## Use Cases

1. Media and Entertainment
2. Public Safety
3. Healthcare
4. Law Enforcement
5. Digital Asset Management
6. Influencer Marketing
7. Digital Advertising
8. Education
9. Consumer Storage
10. Geo-location services
11. Gaming
12. Insurance

Almost all industry domains are now innovating with AWS AI<br><br>

## Case Study — Fraud.net

**Fraud.net** is the world's `leading crowdsourced fraud prevention platform`<br>

Fraud.net uses Amazon ML to support its ML models<br>

The company uses **Amazon DynamoDB and AWS Lambda** to run code without
provisioning and managing servers<br>

They use **Amazon Redshift** for data analysis<br>

* Needed to build and train a larger number of more targeted and precise ML models

* Uses Amazon ML to provide more than 20 ML models

* Saves clients $1 million weekly by helping them detect and prevent fraud<br>

> "Amazon ML helps us reduce complexity and make sense of emerging fraud patterns"

— Oliver Clark, CTO, Fraud.net<br><br>


They launch and train ML models in almost half the time it took on other platforms<br>
It reduces complexity and makes sense of emerging fraud patterns<br><br>

## Create an Impact with AI

* Automate manual, effort-intensive processes

* Engage audiences, customers, and employees

* Optimize product quality and customer experiences<br>

To summarize, you can create an impact in your business by automating repetitive and manual
tasks, engaging customers and optimizing product quality using AI<br><br>

***

# 2. What is Machine Learning?
<br>

## In This Video
* Overview
* Use Cases
* Key Concepts
* ML and Smart Applications
* Amazon Machine Learning
* Case Study


## Overview

ML is a subset of AI<br>
It helps you use historical data to make better decisions

ML is also a process where machines take data, analyze it to
generate predictions and use those predictions to make decisions<br>

Those predictions generate results which are used to improve future
predictions<br><br>

### What Can ML Do?

1. Make predictions from huge datasets
2. Optimize utility functions
3. Extract hidden data structures
4. Classify data

This enables a software program to learn and make predictions in the future<br><br>

### The Flywheel of Data

ML enables you to establish a cycle of improvement using the data you collect
from things like clickstreams, purchases, and likes<br><br>

### ML Use Cases

ML is used in a number of ways across a number of industries<br>
e.g. it can be used to detect fraudulent transactions, filter spam emails,
flag suspicious reviews, and so on<br><br>


**A. Fraud Detection**<br>

Mine data ==> Identify patterns and create labels ==> Train model ==> Flag transaction as fraudulent<br><br>


**B. Content Personalization**

> Use predictive analytics models to recommend items

It can also be used to personalize content for users by recommending content and
predictive content loading<br><br>


**C. Targeted Marketing**

> Use prior customer activity to choose the most relevant email campaigns for
  target customers

ML can also be used for targeted marketing matching customers with offers they might like,
choosing marketing campaigns, and cross-selling or upselling items<br><br>


**D. Categorization**<br>

Unstructured content ==> ML model ==> Categorized documents<br>

ML can also be used to automate categorization of documents such as matching
hiring managers and resumes by learning to understand written content<br><br>


**E. Customer service**

> Analyze social media traffic to route customers to customer care specialists

It can be used in customer service to provide predicted routing of customer emails
based on the content and the sender, as well as social media listening capabilities<br><br>

## ML Concepts

Methods and systems that —<br>
* Predict
* Extract
* Summarize
* Optimize
* Adapt

ML systems `discover hidden patterns in data, and use them to predict patterns in the future`<br>

E.g. if you're analyzing retail data and a product name contains words like "jeans" or "jacket",
     then this product category likely belongs to apparel<br>

ML systems learn from examples in the same way that children learn from language or patterns<br>

It can group data into a summary, and it can also define data in a more
granular, concise way<br><br>


Think of ML as a combination of methods and systems<br>
These methods and systems —<br>
* Predict new data based on observed data,
* Extract hidden structure from the data,
* Summarize data into concise descriptions,
* Optimize an action, given a cost function and observed data, &
* Adapt based on observed data<br><br>

### Types of ML Problems

The field of ML is often classified into the following broad categories —
`supervised learning, unsupervised learning, and reinforcement learning`

In **supervised learning**, the inputs to the model, including the example outputs
(a.k.a. labels) are known, and the model learns to generalize the outputs from
these examples

In **unsupervised learning** the labels aren't known.<br>
The model finds patterns and structure from the data without any help

In **reinforcement learning**, the model learns by interacting with its environment
and learns to take action to maximize the total reward<br><br>

### Supervised Learning

1. Teaches the model by providing a dataset with example inputs and outputs

2. Human teacher's expertise is used to tell the model which outputs are correct<br>

*  This doesn't mean that the human teacher has to be physically present, only that
   the teacher's classifications must be present<br>

3. Input ==> Model ==> Output / Prediction

4. Further grouped into classification and regression<br><br>

In supervised learning, the inputs to the model and the example outputs are provided,
& the `model learns to generalize the outputs from these examples`<br>

With the help of a large training dataset, the model learns from its errors and
changes its weights to reduce its prediction error<br>

In `classification`, the **output variable is a category** like color, and it results
in true or false for a particular question<br>

In `regression`, the **output variable is a number or a value** like weight, dollars, or
temperature<br><br>

### Unsupervised Learning

1. No external teacher or pre-trained data

2. Model detects emerging properties in the input dataset

3. Model then constructs patterns or clusters

4. Further grouped into clustering and association<br><br>

In unsupervised learning, a.k.a *self-organization*, there's no teacher<br>
Its based solely on local information<br>

Here, the model uses only the data presented to the network without any labels,
and it detects the emerging properties of the whole dataset<br>

The model then constructs patterns from the available information, without any
pre-trained data<br>

In `clustering`, the model **discovers groupings in the data**, like grouping
customers based on their purchasing behavior<br>

In `association`, the model **discovers rules that govern large chunks of data**,<br>
e.g. customers who buy product A, also tend to buy product B<br><br>

### Reinforcement Learning

In RL, a software agent determines the ideal behavior within a specific context
for a particular problem<br>

The agent takes the input and decides the best action for the problem, then based
on the results of the action, the `agent then receives simple reward feedback
to allow it to learn from its behavior`<br>

The agent is encouraged to select an action that **maximizes the reward in the long-term**<br>

This type of ML algorithm is inspired by behavioral psychology<br><br>

## ML and Smart Applications

Part of getting useful information out of your ML system is having a
smart application<br>

Your smart application will use ML to analyze your data and predict
future outcomes, which are necessary to make business decisions<br>

* Based on what you know about the user — **Will they use your product?**

* Based on what you know about an order — **Is this order fraudulent?**

* Based on what you know about a news article — **What other articles are interesting?**<br><br>

Based on the customer data you already have, you can find patterns in the data and
then generate predictions to drive your product features and improvements<br><br>

### Challenges to Building Smart Applications

3 primary considerations you should take into account when building your ML application —<br>

1. Expertise
2. Scaling
3. Time to operationalize

While ML is a rapidly growing field with an enormous upside for companies to use,
there are some challenges to take into consideration when building your ML-based
smart application<br>

For instance, some ML technology can be complex to use and implement appropriately,
requiring high levels of expertise that can take time to hire or develop<br>

Another challenge is finding the right technology that scales to the needs of
customers<br>

Finally, being able to tie ML to a business application can take time.<br>
In other words, refining your models so that your product app can use that model
productively can require a lot of time<br><br>

## ML on AWS

One way to help address these challenges could be to use **Amazon Machine Learning**<br><br>

### ML Platforms

**Amazon Machine Learning**

* For developers and data scientists who want to focus on building models

**Apache Spark on Amazon EMR**

* Easily create managed Apache Spark clusters from the AWS Management Console, AWS CLI
  or the Amazon EMR API<br><br>

We have offerings in Amazon ML and Spark on Amazon EMR or
Amazon EMR for customers who want to fully manage platform for building models
using their own data<br><br>

For developers and data scientists who want to focus on building models, the
platform services remove the undifferentiated overhead associated with deploying
and managing infrastructure for training and hosting<br><br>


### Supported Predictions for Amazon ML

Amazon ML supports supervised ML approaches<br>
These enable you to predict specific ML tasks such as binary classification,
multiclass classification and regression<br>

* **Binary classification**<br>

  `Predicts the answer to a Yes/No question`<br>
  e.g. is this email spam or not spam?, etc.<br>

* **Multiclass classification**<br>

  `Predicts the correct category from a list`<br>
  e.g. is this product a movie or clothing?<br>
       is this movie a romantic comedy, documentary or thriller?<br>
       OR which category of products is most interesting to this customer?<br>

* **Regression** `predicts the value of a numeric variable`<br>

  e.g. what would the temperature be in Seattle tomorrow?
       how many days before the customer stops using the application?<br><br>

### Building Smart Applications with Amazon ML

At a broad level, these are the steps involved —

1. Train Model
2. Evaluate and optimize
3. Retrieve predictions

To train a model you need to create a data source object pointing to your data,
explore and understand your data, transform data & train your model<br>

To then evaluate and optimize the model, you need to understand model quality
and adjust model interpretation<br>

After that, you can retrieve batch and real-time predictions<br><br>

## Case Study — Zillow

**Zillow** `provides online home information` to tens of millions of buyers and sellers
ever day<br>

> "We can compute Zestimates in seconds, as opposed to hours, by using Amazon Kinesis
   Streams and Spark on Amazon EMR"

— Jasjeet Thind, VP of Data Science and Engineering<br>

**Kinesis** for data ingestion and
**Amazon EMR** for data processing and analysis<br><br>


**Challenges**<br>

* Provide timely home valuations for all new homes
* Perform ML jobs in hours instead of a day
* Scale storage and compute capacity on demand<br><br>

**Solution**<br>

* Runs Zestimate, Zillow's ML-based home-valuation tool on AWS
* Gives customers more accurate data on more than 100 million homes<br><br>

***

# 3. What is Deep Learning?
<br>

DL is a subset of ML, which is a subset of AI<br>

The foundation for the current era of DL was laid in the 80s
and 90s with research from **Yann LeCun** on CNNs, and LSTMs
by **Sepp Hochreiter and Juergen Schmidhuber**<br>

1986 = rediscovery of backpropagation training algo<br>

**Backpropagation algo** `helps the model learn from its mistakes
by leveraging the chain rule of derivatives`<br><br>


In **"Neural Winter"**, research into DL dropped off ==> partly
due to limitations on data and compute<br>

Intro of Internet, smartphones, smart TVs and availability of
inexpensive digital cameras ==> more and more data was available
Computing power was on the rise<br>

CPUs were becoming faster and GPUs became a general-purpose
computing tool

These trends made neural networks progress<br><br>

In 1998, Yann LeCun published a paper on CNNs for image
recognition tasks, but it wasn't until 2007 when the research
began to accelerate again<br>

Advent of GPU + reduction in training time ==> ushered in mainstay
of neural networks and DL<br><br>

DL uses ANNs to process and evaluate its reference data and
come to a conclusion<br>

ANNs are different from traditional compute processing architectures
in that they're designed to operate more like the human brain<br>

DL uses layers of non-linear processing units for feature extraction
and transformation<br>

Each successive layer uses the output from the previous layer
as an input<br>
Algos may be **supervised or unsupervised**<br>

Applications include `pattern analysis` (unsupervised) and
`classification` (could be supervised or unsupervised)<br><br>

Higher-level features are derived from lower-level features
to form a hierarchical representation<br>

Learn multiple levels of representations that correspond to
different levels of abstraction<br>

Traditional ML focuses on feature engineering, DL focuses
on end-to-end learning based on raw features<br>

Each layer is responsible for analyzing complex features in
the data<br><br>


## What is a Neural Network?

A `set of simple, trainable mathematical functions that can be trained to
collectively learn complex functions.`<br>

With enough training data, it can map input data and features
to output decisions<br>

It consists of multiple layers — an input layer, some hidden
layers, and an output layer.<br>

The basic unit of an ANN is artificial neuron = node.<br>

Artifical neurons have several input channels.<br>

A neuron sums these inputs inside of processing stage, and
produces one output that can fan out to multiple other
artificial neurons<br><br>


Input values are multiplied by weights to get weighted value<br>

Then, if appropriate, the node adds an offset vector to the sum
(called **bias**), which adjusts the sum to generate more accurate
predictions based on success or failure of product predictions<br>

Once the inputs have been weighted, then summed, and the bias
is added (if appropriate), the neuron activates if the final value
produced by the preceding steps meets or exceeds the determined
activation threshold ==> Activation step, the final step before
an output is delivered<br><br>

The **Feedforward Neural Network** is `any neural network that
doesn't form a cycle between neurons.`<br>

Data moves from one input to output without looking backwards<br><br>

**Recurrent Neural Network** — NN which looks backwards.<br>

Primary value of RNN comes when `processing sequential information`,
such as text, speech, or handwriting where the ability to
predict the next word or letter is vastly improved if you're
factoring in the words or letters that come before it.<br><br>


**LSTM** approaches revolutionized speech recognition programs.<br>

LSTM is the basis for many of today's `most successful
applications in the speech recognition domain`, text-to-speech
domain, and handwriting recognition domain.<br><br>


**Text analysis use cases** —<br>

1. In finance, social, CRM and insurance domains to name a few
2. Used to detect insider trading, check for regulatory compliance,
brand affinity, sentiment analysis, intent analysis, etc
by essentially analyzing blobs of text<br><br>


**Time series / predictive analysis use cases** —<br>

1. Used in Log analysis, risk fraud detection
2. Used by the supply chain industry for resource planning
3. Used in IoT field for predictive analysis using sensor data
4. Used in social media and e-commerce for building
recommendation engines<br><br>


**Sound analysis use cases** —<br>

1. In security domain for voice recognition, voice analysis
2. In CRM domain for sentiment analysis<br><br>


Deep learning in —<br>

1. Both automotive and aviation industries
where it's used for engine and instrument floor detection
2. In finance industry for credit card fraud detection, etc.<br><br>

**Image analysis use cases** —<br>

1. In security domain --> facial recognition
2. In social media --> tagging and identifying people in pictures<br><br>

## Challenge for Neural Networks — SCALE

**AlexNet** (won the ImageNet competition in 2012)<br>

1. CNN (2012)
2. No. of layers = 8
3. ~650,000 interconnected neurons
4. ~60 million parameters<br><br>

**ResNet-152**<br>

1. CNN using residual learning (2015)
2. No. of layers = 152
3. Millions of neurons and parameters<br><br>

## Deep Learning on AWS

AWS Platform offers `3 advanced deep learning-enabled managed
API services` —<br>

1. Amazon Lex
2. Amazon Polly
3. Amazon Rekognition<br><br>


**Amazon Lex**<br>

1. Service for `building conversational interfaces into any
application` using voice and text.

2. Provides advanced DL functionalities of automatic speech
recognition, converting speech to text, and natural language
understanding to recognize the intent of input

3. Enables users to build highly engaging user experiences, and
lifelike conversational interactions<br><br>

**Amazon Polly**

1. Turns text into lifelike speech, allowing you to create
applications that talk and build entirely new categories of
speech enabled products<br><br>

**Amazon Rekognition**

1. Makes it easy to add image analysis to your applications,
so that your application can detect objects, scenes, and faces in images

2. You can also search and compare faces, recognize celebrities
and identify inappropriate content<br><br>


DL can often be technically challenging requiring you to
understand the math of the models themselves, & the experience
in scaling, training and inference across large distributed
systems<br>

Build custom models using AWS deep learning AMIs<br><br>


## DIY — AWS Deep Learning AMI

1. Built for Amazon Linux and Ubuntu

2. AWS Deep Learning AMIs come pre-configured with
Apache MXNet, TensorFlow, Microsoft Cognitive Toolkit (CNTK),
Caffe, Caffe2, Theano, Torch, PyTorch and Keras

3. Quickly deploy and run any of the above frameworks at scale

4. Support managed auto-scaling clusters of GPU for large-scale
training

* DLAMIs can help you get started quickly

* They're provisioned with many DL frameworks, including
tutorials that demonstrate proper installation, configuration
and model accuracy.

* DLAMIs install dependencies, track library versions, and
validate code compatibility

* Updated every month

* Always have latest versions of the engines and data science
libraries

* No additional charge (pay for what you use)<br><br>


**2 ways to get started with AWS DLAMIs** —<br>

a) One-click deploy of DL compute instance (launch from AWS Marketplace)
Choice of GPUs (large-scale training) and CPUs (running predictions or
inferences)<br>

Both of them give a stable, secure and high-performance
execution environment to run your applications with
pay-as-you-go pricing model<br>

b) Launch AWS CloudFormation template (to train over multiple
instances, use this for a simple way to launch all of your
resources quickly using the DLAMIs)<br><br>


## Case Study (C-SPAN)

**C-SPAN** is a `not-for-profit organization funded by the US cable
industry to increase transparency by broadcasting and archiving
government proceedings.`<br>

They built an automated facial recognition solution which was
slow<br>
They could only index half of the incoming content by speaker
limiting the ability of users to search archived content.<br>

> "We weren't expecting the high degree of facial recognition
accuracy we're getting. It's very exciting, and setting up
Amazon Rekognition was shockingly easy"

— Alan Cloutier, Technical Manager<br><br>

* Integrated **Amazon Rekognition image analysis** to speed
indexing of eight live video feeds

* Uploaded 97,000 images in < 2 hours

* Enables C-SPAN to more than double video indexed — from
3500 to 7500 hours per year

* Reduced labor required to index an hour of video from
60 to 20 minutes

* Deployed in < 3 weeks<br><br>

***

# 4. Understanding Neural Networks

The simplest neural network is a perceptron, a single layer
NN that uses a list of input features, bias term

Activation function (usually non-linear), depends on problem
you're trying to solve e.g. Sigmoid function in binary classification
problems


**Neural Networks** <br>

1. Layers of nodes connected together
2. Each node is one multivariate linear function, with an
univariate nonlinear transformation
3. Trained via stochastic gradient descent
4. Can represent any non-linear function (very expressive)
5. Generally hard to interpret
6. Expensive to train, fast to predict
7. `scikit-learn — sklearn.neural_network.MLPClassifier`
8. Deep Learning frameworks — MXNet, TensorFlow, Caffe, PyTorch<br><br>


**CNN** 

1. Useful for `image analysis`
2. Input is image or sequence image (weighted)
3. For an image, use kernels as filters to extract local features
4. Pooling layer (to reduce the size of output — average and max pooling)<br>
Pooling is a dimensional reduction process
5. Fully connected layer, used to link to the output
6. Output = particular category of the graph or the image it
contains<br><br>


**RNN**<br>

1. Used for time-sharing data, NLP applications
2. Data has sequential or time-sharing features
3. During training process, info flow is not in one direction<br><br>

***

# 5. Machine Learning Algorithms Explained
<br>

## Types of Machine Learning

A system exhibiting intelligent behavior normally needs to
possess two fundamental faculties —<br>

1. Ability to acquire and systematize knowledge<br> 

This is relying on inductive reasoning or coming up with rules
that would explain the individual observations<br>

2. `Inference` = ability to use the acquired knowledge to derive
the truths when needed, like making predictions, choose actions,or
make complex plans<br>

This relies on deductive reasoning<br><br>


**A. Supervised Learning*<br>

There needs to be a supervisor, showing the right answers
during the learning<br><br>


**B. Unsupervised Learning**<br>

No supervisor, only observations or data<br>

E.g. `Clustering algorithms` (divides observations into different
clusters)<br><br>

**C. Reinforcement Learning**<br>

Attempts to solve the complete AI problem of building an agent
capable of exhibiting entire intelligent behaviors, not just
making isolated decisions<br>

Agent controlled by the algorithm is interacting with the possibly
completely unknown environment, learning optimal actions via
trial and error<br>

No teacher telling agent what's the right action at any given
time, instead the agent is getting an `often delayed reward/penalty
called reinforcement`, and is **designed to maximize long-term
rewards**<br><br>

## Supervised Learning

The notion of a "teacher" may be generalized to any complex
system or phenomena that consists of machines, humans, or
natural processes.<br>

Known outcomes form so-called **"ground truths"**<br>

Train the model by feeding it the training dataset<br>

Resulting model is set to predict the same outcome based on
previously unseen input parameters<br>

Reason we're interested in building such models is that
the original system is either impossible or expensive to
procure and scale or takes too long to produce the outcome
which we want to obtain sooner<br><br>


In **multi-class prediction**, data point can belong to one of may
different and mutually exclusive classes<br><br>


If the variable being predicted is numeric, then the model is
said to be solving a `Regression problem` — determining the
unknown value of the dependent variable based on input
parameters

**E.g.** Predictive maintenance, Customer churn prediction,
predicting natural disasters<br>


Tools such as **Amazon Mechanical Turk** used to crowdsource human
decisions<br><br>


### Linear Supervised Algos

Learn parameters of a linear function, likely in a multi-dimensional
space<br>

Find **hyperplane or "decision boundary"** that best separates
the data samples belonging to different classes<br>

Logistic function ==> Outputs are in the range [0, 1]<br>

**Amazon SageMaker** has built-in `"Linear Learner" algorithm` which is
effectively a combination of linear and logistic regression<br>

Other examples are
**Support Vector Machines (SVMs)** which strive to find a
hyperplane with maximum margin b/w classes<br>

**Perceptron** is a simple linear classifier that forms the
foundational unit of ANNs<br><br>


### Non-Linear Supervised Algos

**Decision Trees**<br>

1. In order to make classification, start with root of the tree
and descend through the decision nodes until we arrive at a
classification<br>

Random Forest<br>
XGBoost (general purpose supervised algo)<br>

SageMaker includes the **XGBoost algorithm** based on the idea of
building a strong classifier out of many weak classifiers
in the form of decision trees ==> Boosting<br><br>

**Factorization Machines**<br>

1. Works best when dealing with large amounts of high dimesional sparse data
2. e.g. click prediction for online advertising or recommendations in general
3. Also built into SageMaker<br><br>

**Polynomial** (circular or parabolic boundaries)
**Neural Networks**<br><br>

## Unsupervised Learning Algos

**Clustering**<br>

1. Divide data points into groups or clusters with the
assumption that points belonging to the same cluster are somehow
similar, while those belonging to different clusters are
somehow dissimilar

2. A problem — don't know how many clusters to pick

3. Another problem — ultimately upto us how to interpret
the result and assign meaning to the discovered clusters<br><br>

**Random Cut Forest algorithm**

1. Anomaly detection algo developed by scientists working at
Amazon

2. Works by `constructing a forest of random cut trees.`<br>
Each tree is constructed by a recursive procedure which first
surrounds the data points with a bounding box and then cuts or
splits along the coordinate axis by picking cut points randomly
Procedure's repeated until every point is sorted into a particular
leaf of the tree<br><br>

***Robust Random Cut Forest Based Anomaly Detection on Streams (ICML, 2016)***<br><br>

**Topic Modeling**<br>

1. For documents with text content

2. Basis for the eponymous feature in the Amazon Comprehend service<br>
Given a collection of documents, e.g. news articles, and the
no. of topics we'd like to discover, the algo produces the top
words that appear to define the topic, together with the weight
that each of these words has in relation to the topic<br>

This approach is sensitive to the number of topics requested
and it still requires us to assign the meaning to the discovered
topic<br><br>


SageMaker includes popular clustering algorithm **K-Means**<br>
It's an Amazon improvement over the well-known and scalable
algorithm *web-scale k-means clustering*<br>

SageMaker also includes **Principal Component Analysis (PCA)**<br>

1. Reduces dimensionality within dataset
2. Often used as a feature engineering step before passing the
data to supervised algos<br><br>


**Latent Dirichlet Allocation (LDA)**<br>

1. Particular topic modeling algo
2. A variant is used by the topic modeling feature of Amazon
Comprehend, algo is also available in SageMaker<br><br>


Random Cut Forest algorithm for Anomaly detection is available
in SageMaker and Kinesis Data Analytics for easy application
to streaming data<br>

**Kinesis Data Analytics** also features `"Hot Spot Detection"`,
another example of an unsupervised learning algo used to
identify relatively dense regions in your data<br><br>

## Deep Learning

To a neuron, a data sample is seen as a vector of numeric
input values, which are then linearly combined with its weights<br>

i.e. the Neuron is computing a weighted sum and then applies
an "Activation Function" to produce output in the range from 0 to 1<br>

With proper thresholding, this can work as a binary classifier<br><br>


A single neuron would not be sufficient for practical
classification needs<br>
Instead, we could `combine them into fully-connected layers to
produce ANNs` — **Multilayer perceptrons**<br>

In a **feedforward pass**, the network is turning the input values
into output which forms the prediction of the algorithm<br>

A special technique, **backpropagation** is then used to reduce
the error between the desired or true output and the actual
one produced by the network<br><br>


Originally, these NNs were inspired by some aspects of a biological
nervous system but at this point, they're really a computational
apparatus for complex dependency modeling and function
approximation<br><br>

Since a few years back, we've seen a resurgence of NNs
rebranded as **deep learning** due to several important advances
that relate to the algos themselves.<br>

`Accumulation of large amounts of data for training and
emergence of powerful specialized hardware such as GPUs`, which
are able to crunch this massive amount of data by passing it
through very deep networks in terms of the sheer no. of layers<br>

Some of the results proved rather spectacular enabling many
exciting applications such as in image and speech recognition,
NLP, etc.<br>

Networks with over 1000 layers have been experimented with<br>
Have Billions of parameters and many millions of images could be used in training<br><br>


Sheer computational power required to train such networks is not
cheap to procure<br>

AWS has GPU-based EC2 instances housing powerful chip sets such
as **NVIDIA Volta in the P3 family**<br>

* You can distribute training across multiple GPUs in order
to speed it up; AWS makes it rather economical to set up the
hardware cluster just for the time of training, not having to
worry about expensive hardware sitting idly afterwards<br><br>


### CNNs

One important breakthrough in deep learning was invention of
CNNs, especially `useful for image processing`<br>

It's able to relate nearby pixels in the image instead of
treating them as completely independent inputs, which was the
case prior to CNNs.<br>

Special operation **Convolution** is applied to entire subsections
of the image, the parameters of these convolutions are also being
learned in the process<br>

If several conv layers are stacked one after another, each
conv layer learns to recognize patterns of increasing complexity
as we move through the layers<br><br>

* Recognizing objects and images

* Generally classifying images

* Semantic segmentation (classification of individual
pixels as belonging or not belonging to detected objects)

* Artistic style transfer (one image is modified by applying
an artistic style to it which was previously extracted from
another image, typically a painting)

* Cat image Generator<br><br>


### RNNs

If we `take the output of a neuron and feed it as input to
itself or neurons from previous layers` ==> RNNs<br>

It's as if the neuron remembers its output from the previous
iteration, thus creating a kind of memory<br><br>


**LSTMs** are used for speech recognition and translation<br>

LSTMs are used as a building block in sequence-to-sequence
modeling which is used in neural machine translation<br>

Amazon released an entire library **Sockeye** for `state of the
art seq-to-seq modeling tasks` that customers can use in their
projects<br><br>


**SageMaker provides**<br>

* a built-in algo for image classification, based on ResNet, a kind of CNN
* a sequence to sequence algo (seq2seq),
* a Neural Topic Modeling (NTM) algo to complement LDA
	RNN for text summarization, translation, TTS
* a DeepAR Forecasting algo for time series prediction<br>

Only NTM is unsupervised algo out of the 4 above<br><br>

DL algos have even been employed as a key component of a
reinforcement learning algo<br><br>


***
