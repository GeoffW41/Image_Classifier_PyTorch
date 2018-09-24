import argparse
from utility_functions import load_checkpoint, process_image, process_image_to_tensor, class_predict, parse_idx_to_label
import torch

def main():    
    
    # 1. Getting Input Arguments
    args = get_input_arg()
    # 2. Loading Checkpoint (Trained Model)
    model, model_spec= load_checkpoint(args.checkpoint)
    # 3. Image Preprocessing
    np_img = process_image(args.input)
    # 4. further processing to fit numpy image to the trained model
    img = process_image_to_tensor(np_img)
    # 5. Decide using CPU or GPU for Prediction
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.gpu) else "cpu")
    print(f"Device used for prediction: {device}")
    # 6. Class Prediciton
    df_topk_predict, predicted_class = class_predict(img, model, device, args.top_k)
    # 7. Parse Class inx back to label
    df = parse_idx_to_label(df_topk_predict, args.category_names, model_spec)
    print(f"Top {args. top_k} Prediciton: \n {df}")
 
def get_input_arg():
    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser(description=
                                     """
                                     Make Prediction for an image.
                                     Takes in an image, 
                                     use the trained model loaded from a checkpoint, 
                                     and make the top k prediction based on the classification model.
                                     """, prog = 'predict')
    
    parser.add_argument('--input', type = str, default = 'flowers/test/10/image_07090.jpg', 
                        help = 'Directory for a single image (image)') 
    parser.add_argument('--checkpoint', type = str, default = 'checkpoint_vgg.pth', 
                        help = 'Directory to load trained model (checkpoint)') 
    parser.add_argument('--top_k', type = int, default = 5, 
                        help = 'Returning the top k most likely classes. Takes int. (Default: 5)') 
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', 
                        help = 'Pass in a mapping of categories to real names. Takes str.(Default: cat_to_name.json)') 
    parser.add_argument('--gpu', type = int, default = 1, 
                        help = 'Use GPU for prediciton. Input 0/1.(Default: 1(True))') 

    # Assigns variable args to parse_args()
    args = parser.parse_args()

    # Print Config for Training
    print(f"""
    Image: {args.input} {('(Default)' if args.input == vars(parser.parse_args([]))['input'] else '(User Defined)')}
    Checkpoint: {args.checkpoint} {('(Default)' if args.checkpoint == vars(parser.parse_args([]))['checkpoint'] else '(User Defined)')}
    Top K: {args.top_k} {('(Default)' if args.top_k == vars(parser.parse_args([]))['top_k'] else '(User Defined)')}
    Category name JSON: {args.category_names} {('(Default)' if args.category_names == vars(parser.parse_args([]))['category_names'] else '(User Defined)')}
    GPU Enabled: {bool(args.gpu)} {('(Default)' if args.gpu == vars(parser.parse_args([]))['gpu'] else '(User Defined)')}
    """)
    args.gpu = (bool(args.gpu))
    
    return args
                       
# Call to main function to run the program
if __name__ == "__main__":
    main()