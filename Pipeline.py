# %% [markdown]
# # Pipeline

# %% [markdown]
# ## Table of Contents
# 
# 1. [Preprocessing](#1-Preprocessing)
#     1. [Metadata Extract](#1.A-Metadata-Extract)
#     2. [Frame Extract](#1.B-Frame-Extract)
#     3. [Duplicate Removal](#1.C-Duplicate-Removal)
# 2. [Methods](#2-Methods)
#     1. [Frame Description](#2.A-Frame-Description)
#         1. [Image Captioning (IC)](#2.A.a-Image-Captioning-(IC))
#         2. [Image Segmentation (IS)](#2.A.b-Image-Segmentation-(IS))
#         3. [Object Detection (OD)](#2.A.c-Object-Detection-(OD))
#         2. [Multilabel Classification (MC)](#2.A.d-Multilabel-Classification-(MC))
#     2. [Company-specific Methods](#2.B-Company-specific-Methods)
#         1. [Scene Recognition (SR)](#2.B.a-Scene-Recognition-(SR))
#         2. [Night Recognition (NR)](#2.B.b-Night-Recognition-(NR))
#     3. [Text Recognition (TR)](#2.C-Text-Recognition-(TR))
# 3. [Summarization](#3-Summarization)
# 4. [Evaluation](#4-Evaluation)

# %% [markdown]
# ## Requirements

# %%
# General
import os
import time
from time import strftime
import pandas as pd
import numpy as np
import cv2
import datetime
from datetime import timedelta
from PIL import Image
import torch
import tensorflow as tf
import tensorflow_hub as hub
import torch.nn.functional as F
from transformers import pipeline

### 1 Data Extraction

# 1.A Video Metadata Extraction
import xml.dom.minidom
from deep_translator import GoogleTranslator

# 1.B Video Frame Extraction
import cv2

# 1.C Duplicate Removal
from sewar.full_ref import mse

### 2 Methods

# 2.A Frame Description

# 2.A.a Image Captioning (IC)
image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

# Tokenization
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
nlp = spacy.load("en_core_web_sm")

# 2.A.b Image Segmentation (IS)
from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-tiny-ade")
segmentor = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-tiny-ade")

# 2.A.c Object Detection (OD)
detector = hub.load("https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1").signatures['default']

# 2.A.d Multilabel Classification (MC)
import requests
module = hub.KerasLayer("https://tfhub.dev/google/bit/m-r101x3/imagenet21k_classification/1")

# 2.B Scene Recognition (SR)
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn

# 2.C Night Recognition (NR)

## 2.D Text Recognition (TR)
from paddleocr import PaddleOCR, draw_ocr
paddle_ocr_model = PaddleOCR(use_angle_cls=True,lang='da')
ner = pipeline(task='ner', model='saattrupdan/nbailab-base-ner-scandi', aggregation_strategy='first')

### 3. Summarization
from collections import Counter

# %% [markdown]
# # [1 Preprocessing](#Pipeline)

# %% [markdown]
# ## [1.A Metadata Extract](#Pipeline)

# %%
def extract_metadata(folder_path, xml_file, evaluation=False):
    
    """
    This function takes a path (folder_path) and xml file (xml_file) as inputs and outputs a dataframe containing 
    the id of the scene, the start time of a scene, the end time of a scene and the description of the scene. 
    There are no additional arguments.    
    
    """
    
    if evaluation==True:
        print("[EVALUATION MODE] Extracting Metadata...")
    else:
        print("Extracting Metadata...")
    # Start timer
    start_timer = time.time()
    
    domtree=xml.dom.minidom.parse(folder_path+'/'+xml_file+'.xml')
    group = domtree.documentElement

    # Get duration of video
    duration = datetime.datetime.strptime(group.getElementsByTagName('media_object_group')[0]\
           .getElementsByTagName('mog_duration')[0].childNodes[0].nodeValue, "%H:%M:%S.%f").strftime("%H:%M:%S")
    
    # Extract timecode information
    metadata = group.getElementsByTagName('timecode_description')
    
    # Create empty lists
    scene_id = []
    start_time = []
    descriptions = []
    
    # Looping through the .xml file
    for i, scene in enumerate(metadata):
        # Get the scene-id
        scene_id.append(scene.getAttribute('tid_id'))

        # Get the start-time
        timestamp = scene.getElementsByTagName('tid_timecode')[0].childNodes[0].nodeValue
        timestamp = datetime.datetime.strptime(timestamp, "%H:%M:%S.%f")
        rounded_timestamp = timestamp.strftime("%H:%M:%S")
        start_time.append(rounded_timestamp)

        # Get the description
        try:
            description = scene.getElementsByTagName('tid_description')[0].childNodes[0].nodeValue
        except IndexError:
            description = ''
        descriptions.append(description)
  
    """Only necessary for evaluation phase!"""
    
    # Create a list of abbreviations which shall not be translated and will therefore be removed to later be 
    # added (in their original form) again
    abbreviations = ["Natop. ", "Natopt. ", "Næropt. ","Halvnær opt. ","Supernær opt. ","Ext. opt.", \
                     "Ext. ","Int. opt.","Int. ","Aftenopt.","Natopt.","Natop.","Anonyme opt.","Gen. opt.","Gen. "]
    # Create a ist for all removed abbreviations and the translations (without the removed abbreviations)
    all_removals = []
    raw_translations = []
    
    # Loop through all descriptions
    for sentence in descriptions:
        
        # Create a list for all removed abbreviations for one specific sentence
        removed = []
        
        # Check for all abbreviations (also lower case) if they are contained in a sentence. 
        # If yes, remove them from this sentence and add them to the "removed" list
        for word in abbreviations:
            if word in sentence:
                sentence = sentence.replace(word,"")
                removed.append(word)
            if word.lower() in sentence:
                sentence = sentence.replace(word.lower(),"")
                removed.append(word.lower())
                
        # Add the shortened sentence to the to-be-translated list 
        # and the removed abbreviations to the "all_removals" list
        raw_translations.append(sentence)
        all_removals.append(removed)
        
    # Translate all shortened sentences to english (for comparison)
    raw_translations = GoogleTranslator('da', 'en').translate_batch(raw_translations)
    
    # Exchange all NoneTypes with ""
    raw_translations = ["" if item is None else item for item in raw_translations]
    
    # Add the removed abbreviations back to the beginning of each translated sentence
    descriptions = [" ".join(all_removals[i]) + " " + raw_translations[i] for i in range(len(all_removals))]
    
    # create dataframe and sort values (due to weird structure in DR's .xml files)
    comparison_df = pd.DataFrame({'Scene ID': scene_id, 'Start-Time': start_time, 'Ground Truth Description': descriptions})
    comparison_df = comparison_df.sort_values(by=['Start-Time'], ascending=True, ignore_index=True)

    # create a list for the end time (we need start and end in order to know how to summarize our generated output)
    end_time = []
    
    # get the end-time based on the start-time
    for i in range(len(comparison_df['Start-Time']) - 1):
        start = datetime.datetime.strptime(comparison_df['Start-Time'][i+1], "%H:%M:%S")
        end = start - datetime.timedelta(seconds=1)
        end_time.append(end.strftime("%H:%M:%S"))
    end_time.append(duration)
    
    # Add the end-time to the dataframe
    comparison_df['End-Time'] = end_time
    
    # Re-order the columns of the dataframe and return the dataframe
    comparison_df = comparison_df[["Scene ID","Start-Time","End-Time","Ground Truth Description"]]
    
    # Remove uncommented scenes, scenes describing the whole video (starting with “Background:” or “Description”) 
    # and scenes with incorrectly extracted End-Times (end time of 23:59:59 or earlier than start-time) (only for evaluation) 
    if evaluation == True:
        for row in range(len(comparison_df)):
            if comparison_df["Ground Truth Description"][row] == " " \
            or "Background: " in comparison_df["Ground Truth Description"][row] \
            or "Description: " in comparison_df["Ground Truth Description"][row] \
            or comparison_df["End-Time"][row] == "23:59:59" \
            or (datetime.datetime.strptime(comparison_df["End-Time"][row],"%H:%M:%S")\
                - datetime.datetime(1900,1,1)).total_seconds() \
                - (datetime.datetime.strptime(comparison_df["Start-Time"][row],"%H:%M:%S")\
                - datetime.datetime(1900,1,1)).total_seconds()  < 0:
                comparison_df.drop(row, axis=0, inplace=True)
        comparison_df = comparison_df.reset_index(drop=True)
            
    # Print duration of extraction
    print("Metadata for video '" + str(xml_file) + "' was extracted in " + str(round(time.time()-start_timer,2)) \
          + " seconds. (" + str(round(round(time.time()-start_timer,2)/len(comparison_df),2)) + "s/scene)")
    
    return comparison_df

# %% [markdown]
# ## [1.B Frame Extract](#Pipeline)

# %%
def extract_frames(folder_path,video_id,metadata_dataframe,resize=False,fps_of_video=25,frame_frequency=1, evaluation=False):
    
    """  This function takes a path and a video id (name) as inputs and outputs a dataframe containing the frames """
    """  and their timestamp. Additional arguments are: fps of the video and frame frequency (defining in seconds """ 
    """  in which interval a frame should be extracted).                                                          """
    
    if evaluation==True:
        print("[EVALUATION MODE] Extracting Frames...")
    else:
        print("Extracting Frames...")
    # Start timer
    start = time.time()
    
    # Create two empty lists: one for the frames and one for their timestamps
    frames = []
    timestamps = []
    
    # Create a list of all videos in folder
    video_files = os.listdir(folder_path)
    
    # Filter list for all videos containing this video ID (in case of splits possibly more than 1)
    filtered_list = [name for name in video_files if video_id in name]

    # Creating the folder
    for video in filtered_list:
        video_name = os.path.join(folder_path, video)
        cap = cv2.VideoCapture(video_name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
        interval = fps_of_video*frame_frequency
        
        # Extracting the frames
        for i in range(0, total_frames, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            # Get necessary delta from video title
            if "-" not in filtered_list[0]:
                delta = 0
            else:
                zero_time = datetime.datetime.strptime("00.00.00", "%H.%M.%S")
                duration = video.split("-")[1][:-4]
                duration = datetime.datetime.strptime(duration, "%H.%M.%S")
                delta_duration = duration-zero_time
                delta = delta_duration.total_seconds()
                
            # Get the timestamp of the frame
            if ret:
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                formatted_timestamp = str(datetime.timedelta(seconds=timestamp+delta)) # add delta depending on earlier statement more seconds!
                # Save the frame with the timestamp as the file name
                frames.append(frame)
                timestamps.append(formatted_timestamp)
        
        # Release the video capture object
        cap.release()
    
    # Resize frames if resize factor is defined
    if resize != False:
        frames = [cv2.resize(frame, (round(frame.shape[1]*resize),round(frame.shape[0]*resize)), \
                  interpolation=cv2.INTER_CUBIC) for frame in frames]
    
    # Create dataframe with frames and timestamps
    dataframe = pd.DataFrame({"Timestamp":timestamps,"Frame":frames})
    
    # Remove frames of uncommented scenes (only for evaluation) 
    if evaluation == True:
        # Convert Timestamps of Frames to seconds
        frames_timestamp_seconds = [(datetime.datetime.strptime(dataframe.iloc[row]["Timestamp"],\
            "%H:%M:%S")-datetime.datetime(1900,1,1)).total_seconds() for row in range(len(dataframe))]
        
        # Convert Start and End Times of Metadata to Scene Ranges
        scene_ranges = [range(int((datetime.datetime.strptime(metadata_dataframe.iloc[row]["Start-Time"],"%H:%M:%S")\
                    -datetime.datetime(1900,1,1)).total_seconds()),\
                    int((datetime.datetime.strptime(metadata_dataframe.iloc[row]["End-Time"], "%H:%M:%S")\
                    -datetime.datetime(1900,1,1)).total_seconds()+1))\
                    for row in range(len(metadata_dataframe))]
        
        for row in range(len(frames_timestamp_seconds)):
            if not [1 for scene in scene_ranges if frames_timestamp_seconds[row] in scene]:
                dataframe.drop(row, axis=0, inplace=True)
        dataframe = dataframe.reset_index(drop=True)
            
    # Print amount of extracted frames and duration of extraction
    print(str(len(dataframe))+"(/" + str(len(frames)) + ") frames were extracted in " + str(round(time.time()-start,2)) + " seconds. (" \
          + str(round(round(time.time()-start,2)/len(dataframe),2)) + "s/frame)")
    
    # Store the amount of (extracted) frames we will be working with from now on
    raw_frames = len(dataframe)
    
    return dataframe, raw_frames

# %% [markdown]
# ## [1.C Duplicate Removal](#Pipeline)

# %%
def drop_duplicates(dataframe,mse_threshold=100):
    
    """  
    This function takes a dataframe, a frame column name and a threshold for the MSE measurements as input, 
    filters out all duplicate frames (i.e. one of two frames too similar according to MSE measures, ergo below 
    the given threshold (default=100)) and adds a new column stating the amount of duplicates of one frame.
    """
    
    print("Finding and removing duplicates ...")
    # Start timer
    start = time.time()
    
    # Create a list for counting the duplicates and another for dropping the duplicated frames
    duplicate_counter = []
    drop_indices = []
    
    # Loop through all rows of the dataframe
    for frame in range(len(dataframe)-1):
                
        # Calculate MSE for every frame in comparison to its following (i.e. one second later) frame 
        
        # In case MSE is bigger than threshold, nothing is dropped and the frame's occurence is set to "1"
        if mse(dataframe["Frame"][frame],dataframe["Frame"][frame+1]) >= mse_threshold:
            if frame == 0:
                duplicate_counter.append(1)
                duplicate_counter.append(1)
            else:
                duplicate_counter.append(1)
            
        # If MSE is smaller than threshold, the duplicated (next frame) will be dropped and the duplicate 
        # counter for this frame will be incresed by "1"
        else:
            drop_indices.append(frame+1)
            if frame == 0:
                duplicate_counter.append(2)
            else:
                duplicate_counter[-1] += 1
    
    # Drop all duplicates from dataframe
    dataframe = dataframe.drop(drop_indices)
    
    # Add Counter column to dataframe
    dataframe["Counter"] = duplicate_counter
    
    print(str(len(drop_indices)) + " duplicates (" + str(round(len(drop_indices)/(len(duplicate_counter)\
          + len(drop_indices))*100,2)) \
          + "%) were found and removed in " + str(round(time.time()-start,2)) + " seconds. (" \
          + str(round(round(time.time()-start,2)/(len(drop_indices)+len(dataframe)),2)) + "s/frame)")
    
    unique_frames = len(duplicate_counter)
    
    return dataframe.copy().reset_index(drop=True), unique_frames

# %% [markdown]
# # [2 Methods](#Pipeline)

# %% [markdown]
# ## [2.A Frame Description](#Pipeline)

# %% [markdown]
# ### [2.A.a Image Captioning (IC)](#Pipeline)

# %%
# Function to convert the numpy arrays to a PIL images
def convert_to_pil_image(np_array):
    imageRGB = cv2.cvtColor(np_array, cv2.COLOR_BGR2RGB)
    return Image.fromarray(np.uint8(imageRGB))

# %%
def extract_ngrams(text):
    doc = nlp(text)
    ngrams = []
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text
        # Remove the leading "a" if present
        if chunk_text.startswith('a '):
            chunk_text = chunk_text[2:]
        
        # Remove the leading "the" if present    
        if chunk_text.startswith('the '):
            chunk_text = chunk_text[4:]            

        # Keep only the first adjective and the noun if two adjectives are combined with "and"
        tokens = chunk_text.split(' ')
        if len(tokens) >= 4 and tokens[1] == 'and':
            chunk_text = ' '.join([tokens[0], tokens[3]])

        ngrams.append(chunk_text)
    return ngrams


def tokenize_IC(IC_predictions):

    """
    This function takes a dataframe and a column name as inputs and outputs the dataframe with a tokenized 
    (and adjusted for stopwords) text column.    
    
    """
    
    print("Removing Stopwords and Extracting Noun Chunks...")
    # Start timer
    start_timer = time.time()
    
    # Specifying that we want to drop specific words
    stop_words = set(stopwords.words("english"))
    
    # Create an empty list for tokenized predictions
    tokenized_IC = []
    for row in range(len(IC_predictions)):
        tmp = []
        sentences = sent_tokenize(IC_predictions[row])
        for sent in sentences:
            sent = sent.lower()
            
            # Extract noun chunks (including nouns with their adjectives) using SpaCy
            noun_chunks = extract_ngrams(sent)
            
            # Filter out stopwords
            filtered_noun_chunks = [chunk for chunk in noun_chunks if chunk not in stop_words]
            
            tmp.extend(filtered_noun_chunks)
            
        # Update dataframe with tokenized and adjusted description
        tokenized_IC.append(tmp)

    # Print duration of extraction
    print("Tokenizing Predictions took " + str(round(time.time()-start_timer,2)) + " seconds. (" \
          + str(round(round(time.time()-start_timer,2)/len(dataframe),2)) + "s/frame)")
        
    return tokenized_IC

# %%
def IC(dataframe):
   
    """ This function takes a dataframe and the column name of the frames as input and outputs the same dataframe """
    """     extended with the image caption prediction from the 'nlpconnect/vit-gpt2-image-captioning' model.     """
    
    print("Captioning frames ...")
    # Start timer
    start = time.time()
 
    # Convert all arrays to images
    images = [convert_to_pil_image(frame) for frame in dataframe["Frame"]]
    
    # Predict Frame Captions
    captions = [image_to_text(image)[0]['generated_text'] for image in images] # [0]['generated_text'] to extract captions only
 
    # Tokenize IC predictions and update prediction list accordingly
    tokenized = tokenize_IC(captions)
    
    # Add Predictions to Dataframe
    dataframe["IC"] = tokenized
    
    # Print amount of predicted captions and duration of prediction
    print(str(len(captions)) + " captions were predicted in " + str(round(time.time()-start,2)) + " seconds. (" \
          + str(round(round(time.time()-start,2)/len(dataframe),2)) + "s/frame)")
    
    model1_time = round(time.time()-start,2)
    
    return dataframe.copy(), model1_time

# %% [markdown]
# ### [2.A.b Image Segmentation (IS)](#Table-of-Contents)

# %%
def IS(dataframe,probability_threshold=0.9):
    
    """ This function takes a dataframe as input and outputs the same dataframe """
    """    extended with the image segmentation prediction from the \facebook/maskformer-swin-tiny-ade' model.    """

    print("Segmenting frames ...")
    # Start timer
    start = time.time()

    # create an empty list
    segments = []
    
    # Uses a for-loop to iterate through the files in our folder. 
    for frame in dataframe["Frame"]: # change to dataframe later
        inputs = feature_extractor(images=frame, return_tensors="pt")
        outputs = segmentor(**inputs)
        class_queries_logits = outputs.class_queries_logits
        probabilities_tensor = F.softmax(class_queries_logits, dim=-1)
        
        # Convert the probabilities tensor to a NumPy array
        probabilities_np = probabilities_tensor.detach().numpy()
        
        # Find the unique class indices and their maximum probabilities
        unique_class_indices = list(range(probabilities_np.shape[-1]))
        max_probabilities = probabilities_np.max(axis=(0, 1))
        
        # Zip the class indices and their maximum probabilities together
        class_probabilities = list(zip(unique_class_indices, max_probabilities))
        
        # Filter the class_probabilities to only include those with probabilities higher than 90%
        high_prob_class_probabilities = [(class_idx, probability) for class_idx, probability in class_probabilities if probability > probability_threshold]
        
        # Sort the high_prob_class_probabilities list in descending order of probability
        high_prob_class_probabilities.sort(key=lambda x: x[1], reverse=True)
        
        # Extract the first word of each high-probability class and store them in a new list
        sorted_high_prob_first_words = [segmentor.config.id2label[class_idx].split(",")[0] for class_idx, _ in high_prob_class_probabilities if class_idx in segmentor.config.id2label]
        
        # Store the labels
        segments.append(sorted_high_prob_first_words)

    # Add Predictions to Dataframe
    dataframe["IS"] = segments
    
    # Print amount of predicted captions and duration of prediction
    print(str(len(segments)) + " frames were segmented in " + str(round(time.time()-start,2)) + " seconds. (" \
          + str(round(round(time.time()-start,2)/len(dataframe),2)) + "s/frame)")
    
    model2_time = round(time.time()-start,2)
    
    return dataframe.copy(), model2_time

# %% [markdown]
# ### [2.A.c Object Detection (OD)](#Table-of-Contents)

# %%
def OD(dataframe,probability_threshold=0.1):
    
    """ This function takes a dataframe and the column name of the frames as input and outputs the same dataframe """
    """                        extended with the object detection prediction from the                             """
    """                  'https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1' model.                       """

    print("Detecting objects ...")
    # Start timer
    start=time.time()
    
    # Convert frames to tensors
    converted = [tf.image.convert_image_dtype(tf.constant(frame), tf.float32)[tf.newaxis, ...] for frame in dataframe["Frame"]]
    
    # Detect objects
    prediction = [detector(image) for image in converted]
    results = [{key: value.numpy() for key, value in part.items()} for part in prediction]
    
    # Extract unique classes with their highest probability above the threshold
    filtered_labels = [
        [
            tf.compat.as_str_any(class_entity).lower()
            for class_entity in set(result["detection_class_entities"])
            if max(score for class_entity_iter, score in zip(result["detection_class_entities"], result["detection_scores"]) if class_entity_iter == class_entity) > probability_threshold
        ]
        for result in results
    ]
    
    # Add filtered labels to the DataFrame
    dataframe["OD"] = filtered_labels
    
    # Print amount of predicted captions and duration of prediction
    print(str(len(dataframe)) + " frames were analyzed in " + str(round(time.time()-start,2)) + " seconds. (" \
          + str(round(round(time.time()-start,2)/len(dataframe),2)) + "s/frame)")
    model3_time = round(time.time()-start,2)
    
    return dataframe.copy(), model3_time

# %% [markdown]
# ### [2.A.d Multilabel Classification (MC)](#Table-of-Contents)

# %%
def preprocess_image(image, target_size=(224, 224)):
    # Resize the image
    resized_image = tf.image.resize(image, target_size)

    # Normalize the pixel values to the range [0, 1]
    normalized_image = tf.cast(resized_image, tf.float32) / 255.0

    # Add a batch dimension
    input_image = normalized_image[tf.newaxis, ...]

    return input_image

# %%
def MC(dataframe, probability_threshold=0.01):
    
    """ This function takes a dataframe and the column name of the frames as input and outputs the same dataframe """
    """                        extended with the multilabel classifications from the                              """
    """                  'https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1' model.                       """    
    
    # Print message indicating that label classification has started and record starting time
    print("Classifying Labels ...")
    start = time.time()
    
    # Retrieve list of classes from a text file hosted on a remote server and format class names
    lines = requests.get('https://storage.googleapis.com/bit_models/imagenet21k_wordnet_lemmas.txt').text.split("\n")
    classes = [entity.split(",")[0].replace("_", " ") for entity in lines][:-1]

    # Preprocess each frame in the input dataframe and obtain predictions from a TensorFlow model
    converted = [preprocess_image(frame, target_size=(224, 224)) for frame in dataframe["Frame"]]
    prediction = [tf.nn.sigmoid(module(image)) for image in converted]

    classifications = []
    
    # Iterate over predictions and identify the top classes with probabilities above the threshold
    for probabilities in prediction:
        top_x = []
        prob_list = probabilities.numpy()[0].tolist()
        sorted_results = pd.DataFrame({"classes": classes, "probabilities": prob_list}).sort_values("probabilities", ascending=False)

        for index, row in sorted_results.iterrows():
            if row["probabilities"] > probability_threshold:
                top_x.append(row["classes"])

        classifications.append(top_x)
    
    # Add new column to the dataframe with the predicted classes for each frame
    dataframe["MC"] = classifications
    
    # Print the number of frames classified and the duration of the classification process
    print(str(len(classifications)) + " frames were classified in " + str(round(time.time()-start,2)) + " seconds. (" \
          + str(round(round(time.time()-start,2)/len(dataframe),2)) + "s/frame)")
    
    # Calculate the total time taken for classification and return a copy of the modified dataframe and the time taken
    model4_time = round(time.time()-start,2)
    
    return dataframe.copy(), model4_time

# %% [markdown]
# ## [2.B Company-specific Methods](#Table-of-Contents)

# %% [markdown]
# ### [2.B.a Scene Recognition (SR)](#Table-of-Contents)

# %%
def load_labels():
    """
    The load_labels function reads the category labels from the Places365 dataset and returns them in the form of tuples. 
    It also reads the indoor-outdoor labels, scene categories, and wideresnet model.

    """
    # 1. Prepare all the labels for scene category
    file_name_category = 'categories_places365.txt'
    classes = list()

    # Read the category labels from file
    with open(file_name_category) as class_file:

        # For each line in the file, extract the label name and add to the list
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])

    # Convert the list to a tuple for immutability. By converting the classes list to a tuple, the code ensures that the labels cannot be accidentally modified or changed by other parts of the program.
    classes = tuple(classes)

    # 2. Prepare all the labels for indoor-outdoor
    file_name_IO = 'IO_places365.txt'

    # Read the indoor-outdoor labels from the labels file
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []

        # For each line in the file, extract the label name and add it to the list.
        for line in lines:
            items = line.rstrip().split()
            # subtract 1 to map 1 to 0 and 2 to 1. For the final labels 0 is indoor, and 1 is outdoor.
            labels_IO.append(int(items[-1]) -1) 

    # Convert the list to numpy array for efficiency
    labels_IO = np.array(labels_IO)

    return classes, labels_IO


def returnTF():

    """
    The returnTF function loads the image transformer, which applies a set of transformations 
    (resize, convert to tensor, and normalize) to an image.

    The returnTF function uses the trn.Compose method from the torchvision.transforms module to define a set of 
    transformations that will be applied to an image. 

    """

    # load the image transformer and define a set of transformations to be applied to an image
    TF = trn.Compose([

        # Resize the image to (224, 224)
        trn.Resize((224,224)),

        # Convert the image to a PyTorch tensor
        trn.ToTensor(),

        # Normalize the image with mean and standard deviation
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Return the image transformer
    return TF

def load_model():
    """
    The load_model function loads the pre-trained model, the weights, and hacks the model to handle batch normalization, 
    average pooling, and other module issues in pytorch1.x.

    """
    # this model has a last conv feature map as 14x14

    # Download the model file
    arch = 'resnet18'
    model_file = f'{arch}_places365.pth.tar'

    # import the model architecture
    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)

    # set the model to evaluation mode
    model.eval()

    # register a forward hook on the last convolutional layer (layer4) and the average pooling layer (avgpool) of the model
    features_names = ['layer4','avgpool'] 

    return model


# Load all the labels we need for the scene recognition.
classes, labels_IO = load_labels()

# Create an empty list to store the features extracted from the input image and call it features_blobs
features_blobs = []

#Load the model
places_model = load_model()
#Load the image transformer
TF = returnTF() # image transformer
# Retrieve the parameters of the pre-trained model.
params = list(places_model.parameters())
# Retrieve the softmax weights of the pre-trained model.
weight_softmax = params[-2].data.numpy()
# This sets all negative values in the softmax weight to zero.
weight_softmax[weight_softmax<0] = 0

def SR(dataframe, frame_column_name="Frame"):
    """"
    This function runs the places365 model on the videos.
    Input: video number
    
    PUT IN BY MARIN --> KONRAD CHECK
    
    """
        
    print("Predicting scenes ...")
    # Start timer
    start = time.time()
    
    # Create empty list for predictions
    places = []
    all_scenes = []
    
    for frame in dataframe[frame_column_name]:
        
        # Create an empty list for all predictions
        prediction = []
        
        # Convert array to PIL format
        img = Image.fromarray(frame)
        
        # Pre-process the input image using the image transformer and convert it into a PyTorch variable.
        input_img = V(TF(img).unsqueeze(0))

        # Perform a forward pass through the pre-trained model to obtain the logit values.
        logit = places_model.forward(input_img)
        
        # Apply the softmax function to the logit values and remove the dimensions of size 1.
        h_x = F.softmax(logit, 1).data.squeeze()

        # This sorts the probabilities and the corresponding class indices in descending order.
        probs, idx = h_x.sort(0, True)

        # This converts the probabilities to a NumPy array
        probs = probs.numpy()

        # This converts the class indices to a NumPy array
        idx = idx.numpy()
      
        # This calculates the probability of the input image being indoors or outdoors 
        io_image = np.mean(labels_IO[idx[:10]])
        prediction.append(io_image)
        places.append(prediction[0])
        
        # Create an empty list to store the scene categories which the model predicts
        scene_categories = []
        
        # We say we maximum want 5 scene categories (if they all meet the threshold)
        for i in range(0, 5):
            
            # Define a variable for the classes (scene categories) and for the probabilities of these scene categories.
            category = classes[idx[i]]
            probability = probs[i]
            
            # If the probability is higher than 60%, then add the scene categorie to the list. If not, go on to the next frame.
            if probability >= 0.5:
                scene_categories.append(category)
            else:
                continue
        
        # Add the scene categories per frame to the list that will form the column in the next step
        all_scenes.append(scene_categories) 

    dataframe["SR: Score"] = places
    dataframe["SR: Scene Categories"] = all_scenes
    
    # Print the number of frames classified and the duration of the classification process
    print(str(len(all_scenes)) + " scenes were identified in " + str(round(time.time()-start,2)) + " seconds. (" \
          + str(round(round(time.time()-start,2)/len(dataframe),2)) + "s/frame)")
    
    model5_time = round(time.time()-start,2)
    
    return dataframe.copy(), model5_time

# %% [markdown]
# ### [2.B.b Night Recognition (NR)](#Table-of-Contents)

# %%
def NR(dataframe):
    
    """
    
    This function takes a dataframe as input where the values of the frames are represented in arrays. It accesses the arrays
    and calculates the average pixel value per frame. (Later, in the summarization it will be cehecked if the avg pixel value 
    is below a certain previously defined threshold, and if yes, it determines the frame must be a Natopt)
    It outputs a new dataframe with the timestamp, frames, and a new column with the average pixel values.
    
    """
    
    print("Predicting time of day ...")
    # Start timer
    start = time.time()
    
    # Create an empty list for all predictions
    day_night = []
    
    for frame in dataframe["Frame"]:
        
        # Create empty list for avgerage pixel value
        prediction_day_night = []
        
        # Calculate the average pixel value
        average_pixel_value = np.mean(frame)
        
        day_night.append(average_pixel_value)
    
    dataframe["NR"] = day_night
    
    # Print the number of frames classified and the duration of the classification process
    print(str(len(day_night)) + " scenes were analyzed in " + str(round(time.time()-start,2)) + " seconds. (" \
          + str(round(round(time.time()-start,2)/len(dataframe),2)) + "s/frame)")
    
    model6_time = round(time.time()-start,2)
    
    return dataframe.copy(), model6_time

# %% [markdown]
# ## [2.C Text Recognition (TR)](#Table-of-Contents)

# %%
def TR(dataframe):
    
    print("Detecting text ...")
    # Start timer
    start = time.time()
    
    # Create empty list for the detected text
    tr_results = []
    
    # Apply paddleocr model to recognize text from every frame
    for frame in dataframe["Frame"]:
        
        text = paddle_ocr_model.ocr(frame)[0]
        
        if text:
        
            # Extract text, remove one-letter-words and append to results
            tr_results.append([word[1][0] for word in text if len(word[1][0]) > 1])
            
        else:
            tr_results.append("")
    
    # Named Entity Recognition
    list_ner = [ner(part) if part != "" and part != [] else [] for part in tr_results]
    
    ner_results = []
    for part in list_ner:
        
        # Create an empty list to possibly add no, one or several recognized entities per frame
        interim = []
        for i in range(len(part)):
            
            # If an entity was recognized and it belongs to the LOC or ORG group, add it to the empty interim list
            if part[i] != [] and part[i][0]['entity_group'] in ["LOC","ORG"]:
                interim.append(part[i][0]['word'])
        
        # Add the interim list to the ner_results as list if it contains NERs, otherwise append an empty string
        if interim:
            ner_results.append(interim)
        else:
            ner_results.append("")
                
    # Add the TR and NER results to a new column
    dataframe["NER"] = ner_results
    dataframe["TR"] = tr_results
    
    # Print amount of images with text and duration of prediction
    print(str(len(ner_results)) + " frames were analysed in " + str(round(time.time()-start,2)) + " seconds. (" \
          + str(round(round(time.time()-start,2)/len(dataframe),2)) + "s/frame)")
    
    model7_time = round(time.time()-start,2)
    
    return dataframe.copy(), model7_time

# %% [markdown]
# # [3 Summarization](#Table-of-Contents)

# %%
def summarize_predictions(prediction_dataframe,metadata_dataframe,video_id=False,words_amount=10,threshold=0.1):
    
    """
    This function takes two dataframes and a word amount as arguments outputs an updated metadata dataframe 
    containing the summarized predictions from all different models. The "words_amount" argument defines how
    many words should maximally be in the output and the threshold reduces the output further to only outputting
    words if they were predicted for x% of the frames.    
    
    """
    # Save dataframe of evaluation results as csv
    if video_id != False:
        save = prediction_dataframe.copy()
        del save["Frame"]
        save.to_csv(video_id+"_Predictions.csv",index=False)
        print("Dataframe was successfully saved as '" + video_id + "_Predictions.csv'")
        
    print("Summarizing Predictions...")
    # Start timer
    start_timer = time.time()
    
    # Convert Timestamps of Predictions to seconds
    prediction_timestamp_seconds = [(datetime.datetime.strptime(prediction_dataframe.iloc[row]["Timestamp"],\
        "%H:%M:%S")-datetime.datetime(1900,1,1)).total_seconds() for row in range(len(prediction_dataframe))]
    
    # Convert Start and End Times of Metadata to Scene Ranges
    scene_ranges = [range(int((datetime.datetime.strptime(metadata_dataframe.iloc[row]["Start-Time"],"%H:%M:%S")\
                -datetime.datetime(1900,1,1)).total_seconds()),\
                int((datetime.datetime.strptime(metadata_dataframe.iloc[row]["End-Time"], "%H:%M:%S")\
                -datetime.datetime(1900,1,1)).total_seconds()+1))\
                for row in range(len(metadata_dataframe))]
    
    # Summarize for every model separately
    for model_column in prediction_dataframe.columns:
        
        # A) The Frame Descriptions will be summarized in the same way
        if model_column in ["IC","IS","OD","MC"]:
            
            # Create empty list for all summarizations
            prediction_summarized = []
            
            # Summarize per scene from metadata extract
            for scene in scene_ranges:
                
                # Create empty list for collecting all predicted keywords and counter for threshold computation
                keywords = []
                counter = 0
                
                # Check for every frame if it was in the scene and if so, add its keywords to the pool
                for timestamp in range(len(prediction_dataframe)):
                    
                    # Check if a specific frame belongs to a scene
                    if prediction_timestamp_seconds[timestamp] in scene:
                        
                        # If Duplicates were removed their predictions get higher weight (by their occurrence)
                        if "Counter" in prediction_dataframe.columns:
                            keywords += prediction_dataframe[model_column][timestamp]*prediction_dataframe["Counter"][timestamp]
                            counter += 1*prediction_dataframe["Counter"][timestamp]
                        
                        # If duplicates were not removed no occurence will be considered for the summarization
                        else:
                            keywords += prediction_dataframe[model_column][timestamp]
                            counter += 1
                
                # Adds n(= minimum of all words and "words_amount") words 
                # if they occured more often than a predefined threshold
                prediction_summarized.append(', '.join([Counter(keywords).most_common(min(len(Counter(keywords))\
                                             ,words_amount))[word][0] for word in range(min(len(Counter(keywords))\
                                             ,words_amount)) if Counter(keywords).most_common(min(len(Counter(keywords))\
                                             ,words_amount))[word][1]/counter > threshold]))

            # Create a new column in the metadata dataframe with the summarized predictions for one model
            metadata_dataframe[model_column] = prediction_summarized  
        
        # B) Scene Recognition - Indoor/Outdoor detection
        if model_column == "SR: Score":
          
            # Create empty list for the end result
            int_ext = []
          
            # Summarize per scene from metadata extract. 
            for scene in scene_ranges:
              
                # Create empty list for collecting the i/o score for each frame
                score = []
              
                # For each timestamp within a scene:
                for timestamp in range(len(prediction_dataframe)):
                  
                    # If the timestamp belongs to the scene, add its score to the score list for this particular scene. 
                    if prediction_timestamp_seconds[timestamp] in scene:
                        
                        # If Duplicates were removed their predictions/scores get higher weight (by their occurrence)                        
                        if "Counter" in prediction_dataframe.columns:
                            for duplicate in range(prediction_dataframe["Counter"][timestamp]):
                                score.append(prediction_dataframe[model_column][timestamp])
                      
                        # If duplicates were not removed no occurence will be considered for the summarization
                        else:
                            score.append(prediction_dataframe[model_column][timestamp])
              
                # Calculate average of the score list to find the score for the overall scene.
                average_score = np.mean(score)
              
                # Set the thresholds for when we want to see which output for a scene
                if average_score >= 0 and average_score <= 0.28:
                    int_ext.append('Int. opt.')
                elif average_score > 0.7 and average_score <= 1.0:
                    int_ext.append('Ext. opt.')
                else:
                    int_ext.append("")
                  
            # Later, the int_ext list will be added to the dataframe together with the scene categories list
      
        # Now we move on the the scene categories part of the places365 model
        elif model_column == "SR: Scene Categories":

            # Create an empty list which will later be the foundation for a new column in the dataframe
            location  = []

            # Summarize per scene from metadata extract
            for scene in scene_ranges:
              
                # create empty list for collecting all scene categories
                attributes = []
                counter = 0
              
                # For each timestamp in a scene, check if its part of the scene and then:
                for timestamp in range(len(prediction_dataframe)):
                    if prediction_timestamp_seconds[timestamp] in scene:
                      
                        # If Duplicates were removed their predictions get higher weight (by their occurrence)                        
                        if "Counter" in prediction_dataframe.columns:
                            attributes += prediction_dataframe[model_column][timestamp]*prediction_dataframe["Counter"][timestamp]
                            counter += 1*prediction_dataframe["Counter"][timestamp]
                      
                        # If duplicates were not removed no occurence will be considered for the summarization
                        else:
                            attributes += prediction_dataframe[model_column][timestamp]
                            counter += 1

                # Add the end the division is compared to a threshold 0.5 (hardcoded since by default our threshold is 0.1)
                # But for this specific part of the model we want a threshold of 0.5). We explain in the thesis how we got
                # to this number.
                location.append(', '.join([Counter(attributes).most_common(min(len(Counter(attributes))\
                                     ,words_amount))[word][0].replace("_"," ") for word in range(min(len(Counter(attributes))\
                                     ,words_amount)) if Counter(attributes).most_common(min(len(Counter(attributes))\
                                     ,words_amount))[word][1]/counter > threshold]))
          
            # Later, the scene categories list will be added to the dataframe together with the int_ext list
          
        # C) Night Recognition
        elif model_column == "NR":
          
            # Create empty list for all summarizations
            day_night = []
          
            # Summarize per scene from metadata extract
            for scene in scene_ranges:
              
                # Create empty list for collecting all predicted keywords and counter for threshold computation
                pixel = []
              
                # For each timestamp within a scene:
                for timestamp in range(len(prediction_dataframe)):
                    if prediction_timestamp_seconds[timestamp] in scene:
                  
                        # If Duplicates were removed their predictions/scores get higher weight (by their occurrence)                        
                        if "Counter" in prediction_dataframe.columns:
                            for duplicate in range(prediction_dataframe["Counter"][timestamp]):
                                pixel.append(prediction_dataframe[model_column][timestamp])
                        
                        # If duplicates were not removed no occurence will be considered for the summarization
                        else:
                            pixel.append(prediction_dataframe[model_column][timestamp])
                  
                # Calculate the average pixel value per scene
                average_pixel_values = np.mean(pixel)

                # Set thresholds for natopt.
                if average_pixel_values <= 54.15:
                    day_night.append('Natopt.')
                else:
                    day_night.append("")

            # Create a new column in the metadata dataframe with the summarized predictions for one model
            metadata_dataframe[model_column] = day_night   
  
        # D) Text Recognition
        elif model_column == "NER":
         
           # Create empty list for all summarizations which will be the foundations of the new column in the dataframe
            prediction_summarized_ner = []
          
            # Loop through all scenes in a video
            for scene in scene_ranges:
              
                # Create necessary empty lists to collect results
                text_ner = []
              
                # Go through each frame (per scene)
                for timestamp in range(len(prediction_dataframe)):
                    if prediction_timestamp_seconds[timestamp] in scene:
      
                        # Duplicate Removals are not considered as all named entities will be extracted either way
                        if prediction_dataframe[model_column][timestamp] != "":
                            text_ner += prediction_dataframe[model_column][timestamp]
              
                # Add the unique results to the predefined list
                prediction_summarized_ner.append(', '.join([word[0] for word in Counter(text_ner).most_common()]))            
              
                # Later, the prediction_summarized_ner list will be added to the  
                # dataframe together with the prediction_summarized_no_ner list
          
        elif model_column == "TR":
            prediction_summarized_no_ner = []
          
            # Loop through all scenes in a video
            for scene in scene_ranges:
              
                # Create necessary empty lists to collect results
                text_no_ner = []
                counter = 0
              
                # Go through each frame (per scene)
                for timestamp in range(len(prediction_dataframe)):
                    if prediction_timestamp_seconds[timestamp] in scene:
                      
                        # Remove all words from prediction which were already recognized as NERs to avoid duplicates
                        cleaned_tr = [phrase for phrase in prediction_dataframe[model_column][timestamp] \
                                      if phrase not in prediction_summarized_ner[scene_ranges.index(scene)]]
  
                        # If Duplicates were removed their predictions get higher weight (by their occurrence)
                        if "Counter" in prediction_dataframe.columns:
                            text_no_ner += cleaned_tr*prediction_dataframe["Counter"][timestamp]
      
                        # If duplicates were not removed no occurence will be considered for the summarization
                        else:
                                                      
                            if cleaned_tr != "":
                                text_no_ner += cleaned_tr
              
                # For the rest of the text that was not detected by NER, sort words (descending) by their frequency and add 
                # them (uniquely) to the list. Here, no further threshold will be applied (since the word count filter will 
                # be applied in a later step for the whole 'Text Recognition' column).
                prediction_summarized_no_ner.append(', '.join([word[0] for word in Counter(text_no_ner).most_common()]))
          
    # Merge the two text recognition lists together and add them to the dataframe (if both exist, connect them with a comma)   
    metadata_dataframe['TR'] = [prediction_summarized_ner[i] + ', ' + prediction_summarized_no_ner[i] \
                                if prediction_summarized_ner[i] and prediction_summarized_no_ner[i] \
                                else prediction_summarized_ner[i] + prediction_summarized_no_ner[i] \
                                for i in range(len(metadata_dataframe))]

    # Merge the two scene recognition lists together and add them to the dataframe (if both exist, connect them with a space)   
    metadata_dataframe['SR'] = [int_ext[i] + ' ' + location[i] if int_ext[i] and location[i] \
                                else int_ext[i] + location[i] for i in range(len(metadata_dataframe))]
  
    # Filter all summarizations for the first x words:
    def get_first_x_words(text, x):
        return ', '.join(text.split(',')[:x])
    # 1) IC: x words:
    metadata_dataframe['IC'] = metadata_dataframe['IC'].str.strip().apply(get_first_x_words, x=6)
    # 2) IS: x words:
    metadata_dataframe['IS'] = metadata_dataframe['IS'].str.strip().apply(get_first_x_words, x=6)
    # 3) OD: x words:
    metadata_dataframe['OD'] = metadata_dataframe['OD'].str.strip().apply(get_first_x_words, x=4)
    # 4) MC: 4 words:
    metadata_dataframe['MC'] = metadata_dataframe['MC'].str.strip().apply(get_first_x_words, x=4)
    # 5) SR: Since Scene Categories are already strongly restricted, no further filters will be applied
    # 6) NR: No further filters needed
    # 7) TR: 10 words:
    metadata_dataframe['TR'] = metadata_dataframe['TR'].str.strip().apply(get_first_x_words, x=10)
    
    # Print duration of summarization
    print("Summarization of " + str(len(scene_ranges)) + " Scenes for " + str(len(["yes" for column in \
          list(prediction_dataframe.columns) if "Model" in column])) + " Models took " + \
          str(round(time.time()-start_timer,2)) + " seconds. (" \
          + str(round(round(time.time()-start_timer,2)/len(prediction_dataframe),2)) + "s/frame)")
    
    # Add a new column to the dataframe with the video id
    metadata_dataframe.insert(0,"Video ID",video_id)
    
    # Save dataframe of evaluation results as csv
    metadata_dataframe.to_csv(video_id+"_Summarizations.csv",index=False)
    print("Dataframe was successfully saved as '" + video_id + "_Summarizations.csv'")
    
    return metadata_dataframe.copy()

# %% [markdown]
# # [4 Evaluation](#Table-of-Contents)

# %%
def evaluate_predictions(dataframe):
    
    """
    This function takes one dataframe as input and adds quality scores 
    (of the predictions to the ground truth) to the output dataframe.
    
    """
    
    print("Evaluating Predictions...")
    # Start timer
    start_timer = time.time()
    
    # Create list and dictionary with empty lists as values of relevant columns
    list_of_models = [column for column in dataframe.columns if column[:2] in ["IC","IS","OD","MC","SR","NR","TR"]]
    dict_of_models = {column:[] for column in dataframe.columns if column[:2] in ["IC","IS","OD","MC","SR","NR","TR"]}

    for scene in tqdm(range(len(dataframe))):
        
        # Compute the Embedding for the Ground Truth
        ground_truth_embedding = sentence_transformer.encode(dataframe["Ground Truth Description"][scene])
        
        # Compute Embeddings for all Predictions
        predictions_embedding = sentence_transformer.encode([dataframe[column][scene] for column in list_of_models])
        
        # Compute the Dot Score of every Prediction to the Ground Truth
        scores = util.dot_score(ground_truth_embedding, predictions_embedding)[0].tolist()
        
        # Append Scores to Model List
        for model_name in range(len(list_of_models)):
            # Avoid unexpected high evaluation scores in case of no prediction
            if dataframe[list_of_models[model_name]][scene] == "":
                dict_of_models[list_of_models[model_name]].append(0)
            else:
                # Only add absolute values as negative equal to positive ones but distort mean calculation
                dict_of_models[list_of_models[model_name]].append(abs(scores[model_name]))
    
    # Append every Model Score List as new Column to the given dataframe
    for item in dict_of_models:
        dataframe["".join([item.split(":")[0],"_Score"])] = dict_of_models[item]
    
    # Print duration of summarization
    print("Evaluation of " + str(len(dataframe)) + " Predictions from " + str(len(list_of_models)) \
          + " different Models took " + str(round(time.time()-start_timer,2)) + " seconds. (" \
          + str(round(round(time.time()-start_timer,2)/len(dataframe),2)) + "s/scene)")
    
    return dataframe.copy()


