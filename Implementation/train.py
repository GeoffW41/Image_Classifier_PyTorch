"""
train.py: for training a new network
"""
import argparse

import torch
from model_functions import load_dataset, select_model, build_classifier_model
from train_model import train_network, validation

# Main program function defined below
def main():
    # 1. Getting Input Arguments
    args = get_input_arg()
    # 3. Loading Datasets
    trainloader, validloader, testloader, dataset_info = load_dataset(args.dir)
    # 4. Loading the Pre-trained Models
    model = select_model(args.arch)
    # 5. Defining a classifier networks to be trained
    model, model_spec = build_classifier_model(model, args.hidden_unit, args.drop_p, args.arch, dataset_info)
    # 6. Training the model
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.gpu) else "cpu")
    train_network(model, device, trainloader, validloader, args.epoch, args.save_dir, model_spec, args.testmode)
    # 7. Validating the Model
    test_loss, accuracy = validation(model, testloader, model_spec['criterion'], device)
    print(f"Accuracy on Test Set: {accuracy * 100 :.2f}%")
    
  
def get_input_arg():
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser(description='Traing a Neural Network.', prog = 'train')
    
    parser.add_argument('--dir', type = str, default = '/home/workspace/aipnd-project/flowers', 
                        help = '(str) Dirctory for the training images.') 
    parser.add_argument('--save_dir', type = str, default = '/home/workspace/aipnd-project', 
                        help = '''(str) 
                        Directory to save training checkpoints.
                        checkpoint will be saved with arch name.
                        ''') 
    parser.add_argument('--arch', type = str, default = 'vgg', 
                        help = """
                        (str) Model for training (Default: vgg), 
                        Select from 'vgg', 'densenet' or 'renset' 
                        """) 
    parser.add_argument('--learning_rate', type = float, default = 0.001, 
                        help = '(float) Learning Rate (Default:0.001)') 
    parser.add_argument('--epoch', type = int, default = 10, 
                        help = '(int) No. of Epochs') 
    parser.add_argument('--hidden_unit', nargs='+', type = int, default = [1024, 256], 
                        help = '''
                        (1 or more int) No. of hidden units.
                        Example: [--hidden_unit 256 128 64]
                        ''') 
    parser.add_argument('--drop_p', type = float, default = 0.5, 
                        help = '(float) Drop out rate within classifier layers (Default:0.5)')
    parser.add_argument('--gpu', type = int, default = 1, 
                        help = '(0/1) Use GPU for training (Default: True)') 
    parser.add_argument('--testmode', type = int, default = 0, 
                        help = """
                        (0/1) Test mode: train with 1 batch/epoch only.
                        no checkpoint will be saved (Default: False)
                        """) 

    # Assigns variable args to parse_args()
    args = parser.parse_args()

    # Print Config for Training
    print(f"""
    Dataset Directory: {args.dir} {('(Default)' if args.dir == vars(parser.parse_args([]))['dir'] else '(User Defined)')}
    Checkpoint Saveing Location: {args.save_dir} {('(Default)' if args.save_dir == vars(parser.parse_args([]))['save_dir'] else '(User Defined)')}
    Model for Training: {args.arch} {('(Default)' if args.arch == vars(parser.parse_args([]))['arch'] else '(User Defined)')}
    Learning Rate: {args.learning_rate} {('(Default)' if args.learning_rate == vars(parser.parse_args([]))['learning_rate'] else '(User Defined)')}
    No. of Epoch: {args.epoch} {('(Default)' if args.epoch == vars(parser.parse_args([]))['epoch'] else '(User Defined)')}
    No. of Hidden Units: {args.hidden_unit} {('(Default)' if args.hidden_unit == vars(parser.parse_args([]))['hidden_unit'] else '(User Defined)')}
    GPU Enabled: {bool(args.gpu)} {('(Default)' if args.gpu == vars(parser.parse_args([]))['gpu'] else '(User Defined)')}
    Test Mode: {bool(args.testmode)} {('(Default)' if args.testmode == vars(parser.parse_args([]))['testmode'] else '(User Defined)')}
    """)
    
    args.gpu = (bool(args.gpu))
    args.testmode = (bool(args.testmode))
    
    return args
                       
# Call to main function to run the program
if __name__ == "__main__":
    main()