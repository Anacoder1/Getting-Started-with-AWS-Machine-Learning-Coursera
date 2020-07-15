☀ [1. Introduction to Amazon Comprehend](https://github.com/Anacoder1/Getting-Started-with-AWS-Machine-Learning-Coursera/blob/master/Week%204/Detailed%20Notes.md#1-introduction-to-amazon-comprehend)<br>

☀ [2. Introduction to Amazon Comprehend Medical](https://github.com/Anacoder1/Getting-Started-with-AWS-Machine-Learning-Coursera/blob/master/Week%204/Detailed%20Notes.md#2-introduction-to-amazon-comprehend-medical)<br>

☀ [3. Introduction to Amazon Translate](https://github.com/Anacoder1/Getting-Started-with-AWS-Machine-Learning-Coursera/blob/master/Week%204/Detailed%20Notes.md#3-introduction-to-amazon-translate)<br>

☀ [4. Introduction to Amazon Transcribe](https://github.com/Anacoder1/Getting-Started-with-AWS-Machine-Learning-Coursera/blob/master/Week%204/Detailed%20Notes.md#4-introduction-to-amazon-transcribe)<br>

☀ [5. Deep Dive with Amazon Transcribe](https://github.com/Anacoder1/Getting-Started-with-AWS-Machine-Learning-Coursera/blob/master/Week%204/Detailed%20Notes.md#5-deep-dive-with-amazon-transcribe)<br><br>

# 1. Introduction to Amazon Comprehend
<br>

## In This Course
1. Service introduction
2. Overview and use cases
3. Demonstration
<br>

## Why Amazon Comprehend?

* Unstructured text info is growing exponentially
* Valuable insight is locked within this text
* AI is enabling solutions to analyze text with human-like context
* Enabling NLP accessibility for everyone<br><br>


Text analytics and NLP specifically has been around for a while,
but it's really been rules-based allowing you to parse
unstructured data, so you can do things like keyword counting
and sorting.<br>

Now with DL models, we're able to train this technology to
bring human-like context and awareness<br>


Comprehend offers `5 main capabilities (all powered by DL)` —<br>

1. Sentiment
2. Entities
3. Languages detection
4. Key phrases
5. Topic modeling<br><br>


**Sentiment** allows you to `understand whether what the user is saying
is positive or negative, or even neutral`, sometimes that's important as well.<br>

You want to know if there's not sentiment, that might be a signal.<br><br>

**Entities** — This feature goes through the unstructured text and extracts
            entities and actually categorizes them for you.<br>

So things like people, or things like organizations will be given a category<br><br>

**Language Detection**<br>

For a company that has a multilingual application, with a multilingual
customer base, you can actually determine what language the text is in -->
so you can translate the text or take some other business action on it.<br><br>

**Key Phrases**<br>

Think of this as noun phrases, so where entities are extracted it may be
proper nouns; the key phrase will catch everything else from the unstructured
text, so you can actually go deeper into the meaning. <br>
E.g.<br>
--> What were they saying about the person?<br>
--> What were they saying about the organization?<br><br>

**Topic Modeling**<br>
  It `works over a large corpus of documents and helps you do things like
  organize them into the topics contained within those documents.`<br>
  
* It's really nice for organization and information management

We've actually brought topic modeling as a service<br>

* Topic Modeling as a service
* Extract up to 100 topics from a corpus of documents
* Automatically organizes documents into the topics<br><br>

Topic Modeling is based on an the **Latent Dirichlet Allocation (LDA)** algo.<br>
It's hard to go set up — you have to find an environment, there are lots
of parameters to tune, you have to deploy and operate that environment, to
run that algorithm<br>

Topic Modeling, thanks to Amazon, is available as a simple API; think of TM
as a service<br>

**Topic** — Keyword bucket, so you can see what's in the actual corpus of documents themselves<br>

The service also returns to you an automatic view which maps documents to the topics.<br>
  
E.g. you can take a 1000 blog posts, understand what's in the blog posts
from a top 100 topic perspective and then actually map all the blog posts
into those topic buckets. So if you wanted to give your users a really
easy way to explore or browse blog posts based on the topics they're
interested in, you could do this with a simple call to this job, and the
job service itself.<br><br>

## Text Analysis

Let's talk a little bit deeper around the APIs that help you with text analysis<br>

"Amazon.com, Inc. is located within Seattle, WA and was founded July 5th, 1994
by Jeff Bezos. Everyone loves the great customer experience they receive."<br>

From the above phrase ==><br>

* **Named Entities**<br>
Organization<br>
Location<br>
Date<br>
Person<br><br>

* **Keyphrases**<br>
Noun-based phrases<br><br>

* **Sentiment**<br>
Positive<br><br>

* **Language**<br>
English<br><br>

## Service Value

1. Accurate
2. Continuously trained
3. Easy to use

**Accurate**<br>
  * We have an engineering team and data science team behind this service
    continually working nonstop to make the service accurate.
    
  * On day one you'll notice that this service is accurate out of the box,
    it's competitive and useful for the accuracy that you need for your
    use cases that you're dependent on.<br>

**Continuously Trained**<br>

  * We have folks collecting data, annotating, training the model, looking
    for accuracy problems, fixing them; we're doing this continuously,
    nonstop
    
  * The more you use this service, the more that you'll be able to have
    the service become accurate for you, based on your own data & then
    based on the fact that the team is training it on your behalf, so the
    service gets better over time<br>

**Easy to use**<br>
  * As opposed to understanding what a model is, thinking about training
    or invoking a model, you can simply walk up and it's included in the
    AWS SDK. You can simply invoke the service, it's a REST API
    
  * You could build the service in conjunction with an AWS analytics
    service quite easily<br><br>

## Common Patterns

What are we hearing from our customers, around where do they want to get
started with their NLP solutions?<br>
The patterns `boil down to these 3 areas —`<br>

I. **Voice of Customer Analytics**<br>
II. **Semantic Search**<br>
III. **Knowledge Management / Discovery**<br>

I.<br>
`What are your customers, what is anyone really generally saying about
your brand product or service?`<br>

* Really important in understanding if the new product you've launched,
how are people perceiving it? Do they like the price, etc.<br>
  
* This can be from social media, from comments they're leaving on a site
somewhere, emails they're sending to your company, could be even support
conversations that your agents are noting within support call notes<br><br>

II.<br>
E.g. you're an Elasticsearch customer and you're currently indexing a
corpus of documents to make them available to users. You can use the
NLP service to extract things like topics, key phrases and entities,
& index on those as well<br>

* Your customers can get a better natural search experience<br>
You could suggest other documents from the search experience based
on topics contained within the search result<br>

It just makes search better, understanding what's in the documents
themselves outside of just a keyword context<br><br>

III.<br>
We hear a lot of customers say I want to take a big corpus and organize
them, I want to understand what's in these documents, I've got a variety
of use cases from making this document corpus easier to navigate all the
way to really looking for what's contained in these documents
to make sure that we're meeting certain standards around what info can be
stored in documents<br><br>

## Social Analytics Solution

> Analyze social media postings and comments to organize and classify
customer feedback and look for common patterns.

1. Customers tweeting about brand service on Twitter

2. We've set up a **Kinesis Firehose** which is calling the Twitter
search API, and it's pulling in tweets that we've set to filter
out, that we think is pertinent to us

3. We're then running those tweets through the NLP service to
extract things like the entities in the tweets, the sentiment or
even the key phrases in the tweets.<br>
We might even determine the language these tweets are in, so we
really understand more about where our customer base is in the
world and what they're saying<br>

4. We could store those tweets into a relational service in this case,
or we could use **Amazon S3**

5. We've written all the output from the NLP service into S3, and now
we can just take a query analytics tool like **Athena** & start to
query and analyze the NLP output

6. Once we query that data, we can then build views inside of **Amazon
Quicksight** that shows us things like,<br>
who's mentioning other organizations when they're tweeting about my brand?<br>
Who's mentioning my brand / service / product in a negative context, and why?<br>
What are the keywords, key phrases that they're using when they talk about my brand?<br>

This could allow us to do a variety of things like, customers in a specific
part of the world are interpreting the product we've launched as maybe too
expensive<br>

Bringing the NLP service together with AWS analytics capabilities allows you
to really do text analytics at scale for a wide variety of scenarios<br><br>

***

# 2. Introduction to Amazon Comprehend Medical
<br>

## What is Amazon Comprehend Medical?

*  It's a set of `HIPAA-eligible ML powered APIs built specifically for the
   healthcare domain`
*  It makes it easy to extract and structure info from unstructured medical
   text
*  It does this with state-of-the-art accuracy, helping developers build
   applications that can improve patient outcomes<br>

Comprehend Medical makes advanced medical text analytics accessible to
all developers with no upfront costs<br><br>

## What problems exist that Comprehend Medical can solve?

> 90% of healthcare providers use electronic health records to store
patient data.

Although there are few structured fields in an EHR, most of the
valuable patient-care info is trapped in a free form clinical text
e.g. admission notes, patient history, discharge notes<br>

However, extracting value to find insights from unstructured clinical
notes is a manual and labor-intensive process, creating a bottleneck
that slows down analysis that could result in better health and
business outcomes<br><br>

## How does Comprehend Medical Work?

It's an `extension of Amazon Comprehend's NLP models for entity extraction
of medical texts`<br>

It uses DL to extract entities from unstructured text in the healthcare
field, such as clinical notes and radiology readings<br>

Comprehend leverages the latest advancements in ML to bring a high level
of accuracy and efficiency to extracting clinical information<br>

Comprehend Medical `consists of 2 APIs` —<br>

1. **NERe API** = returns a JSON with all the extracted entities, their traits, and the relationships between them

2. **PHId API** = returns just the protected health information contained in the text<br><br>

Developers can easily integrate Comprehend Medical into their data processing
pipelines with tools like **Amazon Glue**<br>

They can also access it from SageMaker and extract structured data to build
accurate models for healthcare use cases<br>

Once the text is extracted, it can be stored in services like **S3, Aurora,
RDS and Redshift** or any third party service<br><br>

## What does this mean for you?

==> Helps you potentially<br>

* Improve outcomes
* Provide research insights
* Reduce Costs<br>

Identifying a high-risk patient on time will prevent further complications for
the patient and reduce the financial costs for the health system<br>

This data is also valuable for use cases such as clinical decision support,
revenue cycle management and clinical trial management & is difficult to use
without significant manual effort<br>

The ability to extract and structure info from unstructured medical text with
state-of-the-art accuracy no longer requires you to be a medical or ML expert<br><br>


Comprehend Medical makes advanced medical text analytics accessible to all
developers with no upfront costs<br>

CM's current performance has been better than what we have seen in academic
benchmarks<br><br>


## Demo

The color is specific to a different entity type that we extract —<br>

* Orange = Protected health information
* Green = Medical condition<br><br>


We have some entity traits e.g. negation (if patient denies taking some medication,
that medication would be negated)<br>

We show whether a diagnosis is a sign or a symptom<br>

* It's important because when you work downstream it's helpful to have that
differentiation in order to fit into a lot of workflows that exist with our
healthcare customers<br>

What we also do with relationship extraction is we tie the subtypes to the
parents<br>

We tie the medication to the dosage, route or mode so you can make simple
searches<br>

* Provides confidence scores for entity extraction
* You can sift and sort through the data as you deem fit
* Subtypes can also be nested in relationship extraction; so when you put
    this data into Dynamo or Redshift, it will be easier to search<br><br>

JSON output looks like this<br>

```
{
  "Entities": [
    {
      "Id": 18,
      "BeginOffset": 6,
      "EndOffset": 21,
      "Score": 0.9381155967712402,
      "Text": "LABAHN HOSPITAL",
      "Category": "PERSONAL_IDENTIFIABLE_INFORMATION",
      "Type": "ADDRESS",
      "Traits": []
    },
  ]
}
```
<br>

We believe that if we can do a lot of the heavy lifting and hard work, we can
actually enhance our customers to build really cool applications that can
really change and impact healthcare<br><br>

***

# 3. Introduction to Amazon Translate
<br>

## In This Course
1. Service Introduction
2. Features and Benefits
3. How it Works
4. How to Get Started
5. Use Cases and Examples


## Introduction to Amazon Translate

1. It's a `neural machine translation service` powered by DL models that allow
   for fast and accurate translation supporting multiple languages.

2. It's a continually trained solution that allows you to perform **batch
   translations** (when you have large volumes of pre-existing text) as well as
   **real-time and on-demand translations** (when you want to deliver content as
   a feature of your application)<br><br>

## Features of Amazon Translate

**I. Secure Communication**<br>

*  Translate offers secure communication b/w the service and your applications
   through SSL encryption
*  Any content processed by Translate is encrypted and stored at rest in the
   AWS Region where you're using the services<br><br>

**II. Authentication and Access Control**<br>

* You can ensure that info is kept secure and confidential by controlling
  access to Translate through AWS IAM policies<br><br>

**III. Integration with AWS Services**<br>

* As an AWS service, Amazon Translate integrates nicely with several other AWS
  services such as<br>
  `Amazon Polly` for translated speech-enabled products,<br>
  `Amazon Comprehend` for analysis of translated text, and<br>
  `Amazon Transcribe` for localized captioning of your media products<br><br>

**IV. Pay-as-you-go**<br>

* With Translate, you only pay for what you use. You're charged based on the total
  number of characters sent to the API for translation.<br><br>

## Benefits of Amazon Translate

**Call the service with just a few lines of code**<br>

1. As a developer, you no longer need to manually extend your applications with
new languages that meet your customer base.<br>
Instead, Translate allows you to create applications that can be used in any
language. And you can do this with only a few lines of code.<br>

**Easy app integration and more efficient data security**<br>

2. If you're already an AWS customer and you're looking for a translation solution,
it's convenient to stay within the AWS ecosystem for easier integration with
other applications and for more efficient security of your data<br>

**Increased accuracy of translation**<br>

3. Amazon Translate, powered by a neural machine translation engine, offers
increased accuracy of translation when compared to traditional statistical and
rule-based translation models.<br><br>

## How It Works

Translate is based on neural networks that have been trained on various language
pairs, enabling the engine to translate b/w 2 different languages<br>

The model is made up of 2 components — `encoder and decoder`<br>

The **encoder** reads the source sentence one word at a time and `constructs a
semantic representation that captures the meaning of the source text`<br>

Translate uses attention mechanisms to understand context and decide which of
those words in the source are most relevant for generating the next target word.<br>

One of the main advantages of the attention mechanism is to enable the
decoder to shift focus on certain parts of the source sentence to make sure
that ambiguous words or phrases are translated correctly<br>

The `decoder` uses the semantic representation and the attention mechanism
to `generate a translation one word at a time in the target language`<br><br>

It may sound complex, but it's all happening under the hood.
Translate takes care of the details for you<br><br>

## Getting Started

Getting started with the service just takes 3 steps —<br>

1. Set up AWS account
2. Create IAM user
3. Connect — Console, AWS CLI, AWS SDKs<br>

First make sure that you have an AWS account and that you've created
and assigned an IAM role with full access to all Amazon Translate API
calls<br>

Then you have 3 ways to connect to Translate — the Management Console,
the AWS CLI, and the AWS SDKs<br><br>

## Connecting from the AWS CLI

If you're going to use the command line to connect to the service, first
make sure you've set up the AWS CLI.<br>
Once that's done, you can use the CLI in 2 ways to translate text with
Amazon Translate<br>

For short text, you can provide the text that you want to translate as a
parameter of the translate-text command<br>

```
aws translate translate-text \
    --endpoint-url endpoint \
    --region region \
    --source-language-code "en" \
    --target-language-code "es" \
    --text "Hello, World"
```
<br>

For longer text, you can provide the source language, target language, and
text in a JSON file<br>

```
aws translate translate-text \
    --endpoint-url endpoint \
    --region region \
    --cli-input-json file://translate.json > translated.json
```
<br>

## Using AWS SDKs

```
import boto3

translate = boto3.client(service_name = 'translate', region_name = 'region',
                         endpoint_url = 'endpoint', use_ssl = True)

result = translate.translate_text(Text = "Hello, World",
                                  SourceLanguageCode = "en",
                                  TargetLanguageCode = "de")

print('Translated Text: ' + result.get('TranslatedText'))
print('SourceLanguageCode: ' + result.get('SourceLanguageCode'))
print('TargetLanguageCode: ' + result.get('TargetLanguageCode'))
```
<br>

## Use Cases

In general, Amazon Translate is the right solution `when you need to
translate high volume of content and you need to do it quickly`<br>

Most use cases fall under one of 2 main categories —<br>

a. Translating web-authored content for localization purposes,
   either on-demand or in real-time<br>
b. Batch translating pre-existing content for analysis and insights<br><br>

## Example — On-Demand Translation

The website is a single-page JavaScript application hosted in a public
S3 bucket and delivered through **Amazon CloudFront**.<br>

The webpage makes REST API calls using **Amazon API Gateway** which invokes
various **Lambda functions**.<br>

These functions trigger Amazon Translate to execute translations<br>
**Amazon Comprehend** analyzes the sentiment of the review<br>
**Amazon Aurora** as the main database of the application<br>

While there's a lot going on here in terms of translation, all of that
was taken care of with just 1 line of Python code in a Lambda function<br><br>

## Example — Chatbot Translation

The app is hosted in **Amazon S3** and delivered through **Amazon CloudFront**<br>

**Amazon Lex** interacts with the user requests for translations<br>

**AWS Lambda** retrieves past translations from **DynamoDB** and requests new
translations, which are provided by Amazon Translate<br><br>

## Example — Batch Translations

This batch of documents is hosted in S3 bucket<br>
To translate, simply indicate the source bucket as well as a target S3
bucket<br>

Working within a limit of 1,000 bytes of UTF-8 characters per request, this
application performs two main functions<br>

1. There's a function that breaks the source string into individual sentences

2. There's the main function which calls the translate operation for each
   sentence in the source string. This function also handles authentication
   with Amazon Translate<br><br>

## Summary

* Neural machine translation engine

* Real-time, on-demand, and batch translations

* Integrates with AWS services

* Connect via the AWS Management Console, AWS CLI, and AWS SDKs<br><br>

Amazon Translate represents the next generation of translation solutions.<br>
It's built on a neural network that leverages DL techniques<br>

Unlike conventional phrase-based machine translation, Translate takes into
account the entire context of the source sentence as well as the translation
it has previously generated.<br>

* This results in more accurate and fluid translation<br>

Amazon Translate is ideal for real-time and on-demand translation of web
and app content that helps you reach a global audience.<br>

It also allows you to perform batch translations of pre-existing text<br>

Translate integrates with a wide variety of other AWS services allowing you to
extend the reach of your applications<br>

The service is easy to get started with and you can access it through the
AWS Management Console, AWS CLI and AWS SDKs.<br><br>

***

# 4. Introduction to Amazon Transcribe
<br>

## In This Course
1. Introduction
2. Features
3. Use Cases
4. Demonstration

**Amazon Transcribe** is an `automatic speech recognition (ASR) service, that's
designed to make it easy for developers to incorporate speech-to-text
capabilities into their applications`<br>

## Then — Structured Data

* Database forms
* Spreadsheets
* Structured documents<br>

Historically, `people saved most data in structured formats that were
relatively easy to search`<br>
e.g. Info entered into databases via forms or saved in spreadsheets is
pretty easy to analyze even at very high volumes.<br><br>


## Now — Unstructured Audio Data

* Phone calls
* Recorded meetings
* Video<br>

Increasingly, we capture data in `less structured formats that are more difficult
to analyze on a large scale.`<br>

In particular, the speech in audio and video files needs to be recorded as text
before it can be searched or categorized<br><br>

## Manual Transcription is Slow

Traditionally, converting audio content to text has required a lot of human
intervention.<br>
Someone has to play back the recording and capture all of the speaker's words
in text form & then review and edit the transcribed content.<br>

Even with the benefit of a speech-to-text tool, a lot of editing and formatting
is required.<br>

This is particularly true for outputs that require high levels of accuracy
and readability, like video subtitles.<br>

The high level of effort to manually convert these media to text means that
only the most critical items are transcribed and valuable information may be
overlooked.<br><br>

## Why Amazon Transcribe?

* Continually trained
* Low- / high- fidelity support
* Accurate
* Readable, helpful transcriptions
* Source-specific accuracy
    Custom vocabulary
    Language selection
    Multiple speaker recognition
* API Interface
* Authentication and Access Control
* Integration with AWS Services<br><br>

Amazon Transcribe can help you address these potential barriers.<br>

**Transcribe** is a `fully managed and continually trained ASR service that
accepts common audio formats including WAV, MP3, MP4, and FLAC, &
can accurately transcribe both low- and high- fidelity audio.`<br>

This makes it a natural fit for use cases such as transcribing customer
service calls into analyzable data or generating subtitles for videos
with a high level of accuracy.<br><br>

Amazon Transcribe uses DL to provide accurate and quick transcriptions.<br>

It was designed to make it easy for you to get even greater accuracy and
usefulness from the transcribed output<br>

E.g. the output is not just one long uninterrupted string of text.<br>
Instead, Transcribe uses ML to add in punctuation and grammatical formatting,
so the texts you get back are immediately usable<br>

Amazon Transcribe also timestamps every word, which makes it easy to align
subtitles for closed captioning.<br>
It also makes it easy to index<br><br>


Transcripts include confidence levels for each word of the transcribed
output, making it easy to determine where more editing or quality
assurance may be required.<br>

Transcribe also helps you mitigate issues with source audio to improve accuracy.<br>

E.g. you can upload custom vocabulary that can help improve the accuracy if the
content you're transcribing has industry-specific terms like medical or legal
terms.<br>

You can choose the language of the source files — the supported languages are
listed, and Amazon Transcribe attributes speech to different speakers to make it
easier to interpret the output<br><br>

Because Amazon Transcribe was designed to be easy for developers to use, there
are only a few steps to get started.<br>
You don't need to understand how the underlying models work and you don't need to
build your own ML models<br>

You invoke the service via API and with a few lines of code, you're ready to
transcribe files on Amazon S3.<br>

You can easily initiate the transcription operation via the AWS Management Console
and you can test the service via the console too<br><br>

Transcribe offers secure communication b/w the service and your applications
through SSL encryption.<br>
You can access your transcriptions via assigned URLs.<br>

You can also make sure that info is kept secure and confidential by
controlling access to Amazon Transcribe through AWS IAM policies<br><br>


The service integrates easily with other AWS services such as **Amazon
Comprehend, Amazon ES, Amazon Translate, and Amazon Athena**.<br>

You can easily leverage powerful DL models combined with highly scalable
index searching and analysis tools to reap high value from your audio and
video content.<br><br>

## Call Center Analysis use case

A call center uses **Amazon Transcribe** to produce transcriptions of
low-fidelity phone calls with high accuracy.<br>

The center uses **Amazon Comprehend** on the transcribed data to identify
key phrases and customer sentiments so that they can classify the calls.<br>

They use **Amazon Athena** to query the data and **Amazon QuickSight** to visualize
the results and analyze trends.<br><br>

## Meeting Transcription use case

The Human Resources department uses Transcribe to generate transcriptions of
meetings and training sessions, then indexes the results using **Amazon ES**.<br>

Internal stakeholders can quickly search for content on-demand to locate topics
of interest<br><br>

## Closed Captioned Video use case

A video production business uses Transcribe to produce transcriptions of
audio tracks for subtitles.<br>

Limited editing is done on the output files and the quality assurance team
can focus on sections that have lower confidence ratings.<br>

Using the timestamps in the transcript file, they can easily align the
captions with the video.<br>

They use **Amazon Translate** to convert the transcript into additional languages<br><br>

## Multi-lingual Conversation use case

You can even translate conversations.<br>
Use **Amazon Transcribe** to generate the transcription of the speaker's message<br>
Use **Amazon Comprehend** to recognize the language of the speaker, and<br>
use **Amazon Translate** to translate the message into the receiver's language<br>

Then, use **Amazon Polly** to read the translated message to the recipient<br><br>

## Getting Started

To try Amazon Transcribe, you'll need —<br>

1. WAV, FLAC, MP3 or MP4 audio file(s) on Amazon S3
2. AWS account with an IAM user that has full access to the Transcribe API calls
3. Familiarity with the CLI and a Text Editor<br><br>

To get started, you'll need to save the file you want to transcribe on Amazon
S3 stored in a bucket with the proper permissions.<br>

The file(s) must be in one of 4 formats — FLAC, WAV, MP3 or MP4, and they must
be less than 2 hours long<br>

For best results, source files should use lossless compression, FLAC or WAV, and
recording should be done in a low-noise environment.<br>
You'll want to limit crosstalk too<br>

You'll need an AWS account with an IAM user who has full access to the
transcribed API calls, and you should be familiar with the CLI.<br>
You'll also need a text editor<br>

We also recommend, that you review the developer's guide which you can find in
the AWS documentation under Amazon Transcribe<br><br>

## Demonstration

* Create a transcription job
* Verify the job's completion
* Review the results

Amazon Transcribe supports sampling rates in the 8,000 - 48,000 Hz range<br>

The service makes transcription results available for 90 days.<br>

The URI of the transcribed file is available for only a few minutes after you
initially request the job results.<br><br>

## Summary

Amazon Transcribe is a fully managed and continually trained ASR service
that is designed to make it easy for developers to incorporate accurate
speech-to-text capabilities into their applications.<br><br>

***

# 5. Deep Dive with Amazon Transcribe

> Improving confidence level with feedback loop

## AI/ML use cases in M&E field

There are couple use cases that you could use with our AI/ML services<br>

E.g. You can use **Rekognition's** `celebrity detection face match` to generate a
sequence of the marker information and you could use that for the video-editing
to help the video editing software<br>

You could use the **Moderation API** to apply a redaction on any kind of unsafe
content, and of course<br>
You also have a **Transcribe service** that could do speech-to-text to generate
subtitles<br>

You can use **Comprehend** to extract the sentiments of the subtitle or you could
use **Translate** to do a multi-language translation as well, to have subtitles in
different language.<br>

You could have all this metadata go into the **Elasticsearch engine**, where you
could generate a `Hot search capability` for all this metadata<br>
E.g. you could search for "flowers" and see all the videos that contain the
metadata on flowers.<br><br>

There are some use cases more tolerant to a low-confidence level of result,
e.g. Hot search<br>

There are also use cases that actually require much higher accuracy and
much higher confidence of the result<br>
e.g. If you're generating a subtitle using **Transcribe**<br>

You can train Transcribe with a feedback loop to get better confidence levels
and accuracy in results<br><br>

## Case study

(Company X) — In-home fitness company; live-streaming, instructor-led
               exercise classes<br>

Need to automate closed caption generation for the classes —<br>

1) To meet US government requirements<br>
2) To minimize operational costs and overhead<br><br>


[Inside Demo]<br>

We're going to transcode this video in order to have a proxy format<br>

Basically we're just extracting the audio track as the video content is of no
use to Transcribe, and we're generating the transcription from this audio
track<br><br>

## Amazon Connect Realtime Customer Audio Transcription

```
"Incoming customer call is going to hit Amazon Connect, and the first thing we're
going to do is quickly play a prompt that says PRESS ONE to start streaming, that's
going to be done by Polly;<br>
Once the customer presses one, we're going to associate the customer audio with
KINESIS VIDEO STREAM and start actually streaming that audio.<br>

The next thing I'm going to do is invoke a LAMBDA function to take the KVS
details like the ARN, the start fragment and TIMESTAMP and so forth
and put that in a DynamoDB table with the customer's phone no. and contact ID<br>

I have a JAVA APPLICATION on my computer that I will then start and the first thing
that the JAVA APPLICATION will do is talk to DynamoDB and retrieve the Kinesis information
based on the phone no. that I have programmed in that application, and then it will
actually go out to Kinesis Video Streams and start consuming that stream & feed it over
to Amazon TRANSCRIBE and take the transcription in real time of the customer audio
and put it out on my CONSOLE. "
```
<br>

All the capitalized words in the transcript above have been provided through
the feedback loop.<br>

We built our own dictionary, adding some extra words to help Transcribe to
better recognize a certain set of words<br>

Recognizing an acronym is a very hard task for Transcribe because they are not
actual words, they're abbreviations.<br>
By default, Transcribe would not be able to recognize an acronym<br><br>

We have a way around that caveat by telling Transcribe to pay attention to
sequences of words<br>

When we're going to create our custom dictionary, we'll tell Transcribe to
pay attention to the sequence of K space V space S.<br>
We're going to spell it that way into the dictionary<br>

This way, Transcribe is going to look through the audio file for this
sequence, and if it matches that sequence, it's going to use the acronym
provided by the custom dictionary<br><br>


We ran several iterations of Transcribe with a different set of vocabulary
in order to improve the average confidence level of the transcription<br>

Another limitation that we have with Transcribe is the `recognition of numbers.`<br>
One way to fix it is to write the number in literals (e.g. "three") so
Transcribe will be able to recognize or identify the number.<br>

Remember, when you're providing a demo to your customer, you need to train
Transcribe in order to get a better result.<br>
Adapt it to the use case of the customer<br>

Training Transcribe doesn't need to happen all the time — Train it at the
beginning and once you reach an acceptable confidence level, you can roll
the service into production and let it work by itself.<br><br>

You'll always need human interaction to complete some of the words that aren't
identified by Transcribe<br><br>

## Architectural View

Upload video to S3 bucket, it triggers a Step Function which calls
StartTranscribeMachine and StartTranscode with MediaConvert<br>

It extracts the audio stream from the video and sends it to Transcribe to
do the transcription<br>

The feedback path is a bit different. It will create a vocabulary dictionary
with Transcribe, then it will rerun the transcription<br>

At the end, we collect the result from Transcribe and do our conversion from the
Transcribe result into the webvtt subtitle<br>

We also use IoT Message Broker here, it provides a publish / subscribe surface
that will actually post all the messages from the backend to IoT and then you'll
have all these web-connected clients that subscribe to the IoT surface<br>

We'll be able to get all these messages asynchronously<br><br>

## Step Functions — Transcoding state machine

The Transcoding state machine does 2 things —<br>

1. It goes through the media converter to extract the audio stream
2. Once the audio stream is extracted, it will send it to the Transcribe
   surface & Transcribe will then generate the transcription result.<br>

StartTranscodeJob .... ==> StartTranscribeStateMachine ... ==> End<br><br>

## Step Functions — Transcribe state machine

When we start the Transcribe state machine, the very first thing it does is to
check whether there is a custom vocabulary or dictionary being provided.<br>

If no, it will just start Transcribe and wait for the transcription to complete.<br>

If yes, it's going to create the vocabulary set in the Transcribe surface,
wait for the vocabulary to complete.<br>
Once this done, it will go back and start Transcribe.<br><br>

## Transcribe — Create vocabulary

It's easy to create custom vocabulary using Lambda functions and Step functions<br><br>


[How to create vocabulary with Node.js]<br>

```
const Basename = 'YOUR_VOCABULARY_NAME';

const vocabularies = (() => {
  const array = (Array.isArray(Vocabularies)) ? Vocabularies : Vocabularies.split(',');
  return array.filter(x => x).map(x => x.trim().replace(/\s/g, '-').toUpperCase());
})();

const transcribe = new TranscribeService({ apiVersion: '2017-10-26' });
const response = await transcribe.listVocabularies({
  MaxResults: 100,
  NameContains: Basename,
}).promise();

const params = {
  LanguageCode: 'en-US',
  Phrases: vocabularies,
  VocabularyName: Basename,
};

if (response.Vocabularies.findIndex(x => x.VocabularyName === Basename) < 0) {
  await transcribe.createVocabulary(params).promise();
} else {
  await transcribe.updateVocabulary(params).promise();
}
```
<br>

* You have to give a vocabulary name, which needs to be unique. If it's not
unique, instead of creating that vocabulary in Transcribe, you'll be updating
it<br><br>

## Transcribe — Start Transcribe service

```
const Basename = 'SOME_AUDIO_FILE';

const params = {
  LanguageCode: 'en-US',
  MediaFormat: 'mp4',
  Media: { MediaFileUri: mediaFileUri },
  /* generate a unique transcribe name */
  TranscriptionJobName: `${Basename}—${new Date().toISOString().replace(/[—:.]/g, '')}`,
};

/* if vocabulary name exists, uses it */
if (Vocabularies) {
  params.Settings = { VocabularyName: Basename };
}
const transcribe = new TranscribeService({ apiVersion: '2017-10-26' });
const response = await transcribe.startTranscriptionJob(params).promise();
```
<br>

## Transcribe — transcription result

```
{
  "jobName": "david-20180615T100204865Z",
  "accountId": "313031290743",
  "results": {
    "transcripts": [{
      "transcript": "..."
    }],
    "items": [{
      "start_time": "0.000",
      "end_time": "0.250",
      "alternatives": [{
        "confidence": "1.0000",    // confidence level
        "content": "Tell"
      }],
      "type": "pronunciation"
    }, {
      "start_time": "0.250",
      "end_time": "0.410",
      "alternatives": [{
        "confidence": "1.0000",
        "content": "us"
      }],...
    }]
  }
}
```
<br>

## Transcribe — webvtt subtitle

```
WEBVTT

1
00:00:00.040 --> 00.00:03.040
Tell us which <c.five>ws</c> services

2
00:00:03.480 --> 00:00:06.480
Absolutely. I love young <c.six>lumber</c> yard, which is a game
```
<br>

**IoT Message Broker**<br>
  Publish / Subscribe surface, used to send a synchronous message from the
  backend to the web client<br>

1. First you create a "thing" by calling AWS IoT create-thing and giving a name<br>

```
# Create a thing
aws iot create-thing --thing-name anyThing
```
<br>

2. Create a IoT policy<br>

```
# Create IoT Policy
aws iot create-policy \
--policy-name anyThingPolicy \
--policy-document \
'{"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Action": "iot:*", "Resource": "*"}]}'
```
<br>

3. Define a message topic<br>

This is the message topic your web clients or your connected clients will subscribe
to, the backend will be posting the message to this particular topic<br>

```
# Define your message topic
anyTopicName/status
```
<br>

4. Attach a policy to Cognito User<br>

In order for the web client and connected clients to subscribe to the topic, you need
to give permission to that connected client<br>

```
# Attach policy to Cognito user (identity)

aws iot attach-policy \
--policy-name anyThingPolicy \
--target "us-east-1:000000-0000..."
```
<br>

## IoT — publisher

```
class IotStatus {
  static async publish(endpoint, topic, payload) {
    try {
      const iotData = new AWS.IotData({ apiVersion: '2015-05-28', endpoint });
      const params = { topic, payload, qos: 0 };
      const response = await iotData.publish(params).promise();
    } catch(e) {
      e.message = `IotStatus.publish: ${e.message}`;
    }
  }
}

/* Backend state machine to publish messages */
const endpoint = 'https://xxxx-ats.iot.eu-west-2.amazonaws.com';
IotStatus.publish(endpoint, 'anyTopicName/status', 'test message');
```
<br>

## IoT — subscriber

* How connected web client will subscribe to the topic

* The IoT SDK does all the heavy lifting for you

* We'll construct an instance of the device, providing the host (Iot endpoint), client's ID, protocol, etc.<br><br>

You'll listen to 2 messages —<br>

1. In 'connect', you subscribe to a particular topic
2. Using 'message', you'll get the message when the backend is posting the
   message to the IoT<br><br>

```
/* Download IoT JS SDK https://github.com/aws/aws-iot-device-sdk-js */

const subscriber = AWSIoTData.device ({
  host: 'xxxxx-ats.iot.us-east-1.amazonaws.com',
  region: 'us-east-1',
  clientId: 'cognito-user',
  protocol: 'wss',    // web socket
  debug: false,
  accessKeyId,
  secretKey,
  sessionToken
});

subscriber.on('connect', () => {
  subscriber.subscribe('anyTopicName/status');
});

subscriber.on('message', (topic, payload) => {
  console.log(`Received: topic ${topic}: payload: ${payload.toString()}`);
  const message = JSON.parse(payload.toString());
  /* do something */
});
```
<br>

## MediaConvert — transcoding

1. Create IAM service role
2. Define a job template
3. Submit job

1. It's important because you need to give permission to the MediaConvert
   surface to have access to our S3 buckets and API Gateway

2. In our case, we're extracting the audio bit stream, so the job templates
   will be audio output only<br><br>

## MediaConvert — IAM service role

```
# IAM role to allow MediaConvert to access S3 / API resources
aws iam create-role \
--role-name MediaConvertServiceRole \
--assume-role-policy-document \
'{"Version": "2012-10-17",
  "Statement": [{"Sid": "", "Effect": "Allow", "Principal": {"Service": "mediaconvert.amazonaws.com"},
                 "Action": "sts:AssumeRole"}]}'

# Attach S3 policy
aws iam attach-role-policy \
--policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess \
--role-name MediaConvertServiceRole

# Attach API Gateway policy
aws iam attach-role-policy \
--policy-arn arn:aws:iam::aws:policy/AmazonAPIGatewayInvokeFullAccess \
--role-name MediaConvertServiceRole
```
<br>

## MediaConvert — define a job template

In this use case, generates 2 files — audio only output (for Transcribe) &
MP4 video output (for playback)<br>

The latter is the proxy file you could play back on the web client<br><br>

## MediaConvert — submit job

```
# First, find out the per-region, per-account endpoint
aws mediaconvert describe-endpoints --region eu-west-1

/*{
  "Endpoints": [{
    "Url": "https://xxxxx.mediaconvert.eu-west-1.amazonaws.com"
  }]
}*/

# submit job to the endpoint
aws mediaconvert create-job \
--role arn:aws:iam::0000000:role/MediaConvertServiceRole \
--settings file://sampleJob.json \
--endpoint https://xxxxxxx.mediaconvert.eu-west-1.amazonaws.com \
--region eu-west-1

/* {
  JSON Job data
}*/
```
<br>

## MediaConvert — Submit job (Node.JS)

```
async function submitJob(template) {
  try {
    /* create a temp instance to find out the endpoint */
    const temporary = new AWS.MediaConvert({ apiVersion: '2017-08-29' });
    const data = await temporary.describeEndpoints().promise();
    const { Endpoints } = data;
    if (!Endpoints || !Endpoints.length) {
      throw new Error('no endpoint found!');
    }
    /* IMPORTANT: create a new instance and use the endpoint from describeEndpoint */
    const [{ Url: endpoint }] = Endpoints;
    const instance = new AWS.MediaConvert({
      apiVersion: '2017-08-29',
      endpoint,
    });
    const response = await instance.createJob(template).promise();
    return response;
  } catch(e) {
    throw e;
  }
}

const template = { /* Job JSON template */ };
const result = await submitJob(template);
```
<br>

## A few gotchas

I. **1 GB file size limit**<br>
Use MediaConvert, Elastic Transcoder, or ffmpeg to extract audio stream from source file<br>

II. **2 hours duration limit**<br>
Split source file and post-process to stitch (modify timecode of)
the transcription result<br>

III. **Working with acronyms**<br>
Put a space in between letters e.g. A W S, E M R, etc.<br>

IV. **Working with numbers**<br>
Pre/post process to convert numbers before and after Transcribe<br>

V. **If accuracy is top priority**<br>
Consider using Amazon Mechanical Turk to outsource the air check (final edit) process<br>

VI. **Create dictionaries (collections of vocabulary) per... instructor, speaker, series of movie**<br><br>

***
