import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


def load_dataset(data_dir):
    '''
    Load Image Data into Model
    Input:
    data_dir - Takes in data following the structure of:
                1. path/train/(int label)/(img.jpg)
                2. path/valid/(int label)/(img.jpg)
                3. path/train/(int label)/(img.jpg)
                
    Return:
    dataloaders: with batch size of 64/32
    dataset_info: with 'class size'(define output layer of model), 'class_to_idx'
    '''
    # Dataset For Training, Validation annd Testing
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Image Transformaiton for Training sets (with Data Augmentation)
    train_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.RandomCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    
    # Image transformaiton for validation and testing sets (without Data Augmentation)
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    
    # Load datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(train_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(train_dir, transform=test_transforms)
    
    # Create dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 32, shuffle=True)
    
    # Get No. of Class and Class Idx in Dataset
    class_size = len(train_data.classes)
    class_to_idx = train_data.class_to_idx
    dataset_info = {'class_size' : class_size,
                    'class_to_idx' : class_to_idx}
    
    
    return trainloader, validloader, testloader, dataset_info


def select_model(arch):
    '''
    Selecting Model for training.
    
    Input:
    arch - user defined model label
    
    Return:
    model - Pretrained PyTorch Model
    '''
    if arch == 'vgg':
        model = models.vgg16(pretrained = True)
    if arch == 'densenet':
        model = models.densenet121(pretrained = True)
    if arch == 'resnet':
        model = models.resnet101(pretrained = True)
    
    return model

def build_classifier_model(model, hidden_units, drop_p, arch, model_info, predict = False):
    '''
    (Training mode)
    Build Classifier Layers of the Model and initialize the weight and bias.
    (Prediction mode) 
    Reconstructed Classifier Layers of the Model from model_spec from checkpoint.
    
    Input:
    model - Pretrained PyTorch Model.
    hidden_units - user defined spec of hidden layers.
    drop_p - dropout probability
    arch - user defined model label
    model_info - either dataset_info (Training mode) or model_spec (Prediction Mode)
    predict - prediction mode ON/OFF
    
    Return:
    model - pre-trained PyTorch Model
    model_spec - dict containing all necesary info for later reconstruting the classifiyer layers
    '''
    
    # Freezing Pre-trained Feature Detectors
    for param in model.parameters():
          param.requires_grad = False
    
    # Exception case for ResNet: Check if classifier module exist in model
    # and get the classifier layer
    # (resnet use 'fc' instead of 'classifier')
    flag_fc = 0
    if 'classifier.weight' in model.state_dict() or 'classifier.0.weight' in model.state_dict():
        module = next(model.classifier.children(), model.classifier)
    else:
        module = next(model.fc.children(), model.fc)
        flag_fc = 1
    
    # Use model specification info if available (prediction mode)
    if predict:
        input_size = model_info['input_size']
        output_size = model_info['output_size']
        hidden_layers = model_info['hidden_layers']
        model_spec = model_info
           
    else:
        # Otherwise, infer input and output sizes from model features
        input_size = module.in_features
        output_size = model_info['class_size']
        hidden_layers = hidden_units.copy()
        
    # Forming a list of Hidden Layers
    hidden_layers.append(output_size)
    layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
    
    # Defining the untrained, feed-forward network, using ReLU activations and dropout
    classifier = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
    for h1, h2 in layer_sizes:
        classifier.extend([nn.ReLU()])
        classifier.extend([nn.Dropout(p=drop_p)])
        classifier.extend([nn.Linear(h1, h2)])
        
    # Log Softmax as output layer
    classifier.extend([nn.LogSoftmax(dim=1)])
    
    # Exception case for ResNet: to use layer name  'fc' instead of 'classifier'
    if flag_fc:
        model.fc = nn.Sequential(*classifier)
        optimizer = optim.Adam(model.fc.parameters(), lr = 0.001)
        print(f'''
        Classifying Layers Summary: 
        {model.fc}
              ''')
    else:
        model.classifier = nn.Sequential(*classifier)
        optimizer = optim.Adam(model.classifier.parameters(), lr = 0.001)
        print(f'''
        Classifying Layers Summary: 
        {model.classifier}
              ''')

    # Setting Training Criterion
    criterion = nn.NLLLoss()
    print('Model constructed.')
    
    # Saving model-related config if model spec not available:
    if not predict: 
        model_spec = {'Model': arch,
                      'input_size': input_size,
                      'output_size': output_size,
                      'hidden_layers': hidden_units,
                      'drop_p' : drop_p,
                      'class_to_idx' : model_info['class_to_idx'],
                      'optimizer' : optimizer,
                      'criterion' : criterion}
    
    return model, model_spec