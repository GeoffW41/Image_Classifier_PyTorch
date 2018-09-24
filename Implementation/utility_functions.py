import numpy as np
import pandas as pd

import torch
from torchvision import datasets, transforms, models
import PIL
import json

from model_functions import select_model, build_classifier_model

def load_checkpoint(filepath):
    '''
    Function to load the trained model
    Input: filepath to the checkpoint.pth
    
    model_spec Dictionary:
    'Model': arch
    'input_size': input_size
    'output_size': output_size
    'hidden_layers': hidden_units
    'drop_p' : drop_p
    'class_to_idx' : model_info['class_to_idx']
    'optimizer' : optimizer
    'criterion' : criterion
    '''
    # Load checkpoint
    checkpoint = torch.load(filepath)
    
    # Load model_spec form check point
    model_spec = checkpoint['model_spec']
    optimizer = model_spec['optimizer']
    
    # Download Model used in checkpoint and rebuilt the model
    model = select_model(model_spec['Model'])
    model, model_spec = build_classifier_model(model, model_spec['hidden_layers'], model_spec['drop_p'], model_spec['Model'], model_spec, predict=True)
    
    # Load Trained Weight and Bias
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_stat_dict'])
    
    print('Trained Data Loaded.')
    
    return model, model_spec
    
    
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = PIL.Image.open(image_path)
    # Obtain image size info
    w,h = image.size
    aspect_ratio = w/h

    # Resizing Image to have shortest eide = 256
    if aspect_ratio < 1:
        new_w = 256
        new_h = int(round(256/aspect_ratio,0))
    elif aspect_ratio > 1:
        new_w = int(round(256*aspect_ratio,0))
        new_h = 256
    else:
        new_w, new_h = 256,256

    image.thumbnail((new_w,new_h))
    
    # Center-cropping Image to size (224,224)
    area = ((new_w-224)/2, (new_h-224)/2, (new_w-224)/2+224, (new_h-224)/2+224)
    cropped_image = image.crop(area)
    
    # Convert to Numpy array
    np_image = np.array(cropped_image)

    # Normalize image and Transpose to (RGB, w, h) format to fit PyTorch Pre-trained Models
    norm_mean = np.array([0.485, 0.456, 0.406])
    norm_std =  np.array([0.229, 0.224, 0.225])
    np_image_norm = ((np_image/255)-norm_mean)/norm_std
    np_image_norm_t = np_image_norm.transpose(2,0,1)
    
    print('Image preprocessed.')
    
    return np_image_norm_t


def process_image_to_tensor(np_image):
    '''
    Pass in an numpy array of image and change it to TEnsor form that fits the model.
    '''
    img = torch.from_numpy(np_image)
    img = img.type(torch.FloatTensor)
    img.resize_(1,3,224,224)
    
    return img
    
    
def class_predict(img, model, device, topk=5):
    '''
    Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # prediction
    with torch.no_grad():
        img, model = img.to(device), model.to(device)

        outputs = model.forward(img)
        
        # obtain top k predicted class
        probs, classes = torch.topk(torch.exp(outputs),topk,dim=1)
        pred = torch.max(torch.exp(outputs),dim=1)[1]
        
        # return list items
        probs, classes, pred = probs.to('cpu'), classes.to('cpu'), pred.to('cpu')
        probs, classes, pred = probs.numpy()[0],classes.numpy()[0], pred.numpy()[0]
        probs = np.round(probs, decimals = 4)
        
        # compose a dataframe for the prediction
        topk_predict = pd.DataFrame({'classes': classes, 'probabilities': probs})
        
    return topk_predict, pred

def parse_idx_to_label(df, category_names, model_spec): 
    '''
    Parse the predicted idx back to true label.
    Input:
    df = Dataframe containing ['classes','probabilities']
    category_names = JSON file containing the label and key
    '''
    # Load JSON
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    # Parse predicted idx back to label
    class_to_idx = model_spec['class_to_idx']
    idx_to_class = dict([ (v, k) for k, v in class_to_idx.items( ) ])
    df['classes'] = df['classes'].map(idx_to_class)
    df['classes'] = df['classes'].map(cat_to_name)

    return df