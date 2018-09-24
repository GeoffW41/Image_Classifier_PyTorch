from workspace_utils import active_session
from time import time
import torch

def validation(model, validloader, criterion, device):
    '''
    Evaluate prediction accuracy of the trained model/ during training.
    Run under [with torch.no_grad():] to prevent altering the weight and bias of the trained model.
    
    Parameters:
    n_test_batch - no. of test batches to test through
                    Optional: reduce n to get faster speed
    
    Input:
    model - trainied classifier model
    validloader - image sets in dataloader form used for evaluating accuracy
    criterion - criterion for Loss calculation
    device - CPU or CUDA
    
    Output:
    test_loss - Error od the model
    accuracy - prediction accuracy
    '''
    # Parameters
    n_test_batch = 5
    
    # Counter reset
    running_loss = 0
    accuracy = 0
    
    # ensure model running on desired device
    model.to(device)
    
    for i in range(n_test_batch):
        images, labels = next(iter(validloader))
        # ensure tensor running on GPU
        images, labels = images.to(device), labels.to(device)
        
        # calculate loss
        outputs = model.forward(images)
        vloss = criterion(outputs, labels)
        vloss = vloss.item() 
        
        running_loss += vloss
        
        # return probability (0-1)
        ps = torch.exp(outputs)
        
        # calculate accuracy
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    # return test loss and accuracy across all images in validation image set
    test_loss = running_loss/n_test_batch
    accuracy = accuracy/n_test_batch
    
    return test_loss, accuracy


def train_network(model, device, trainloader, validloader, epochs, save_dir, model_spec, test_mode=False):
    '''
    Function to train the classifier layers of model.
    
    Parameters:
    epochs - no. of times to loop through all images
    eval_every - no. of batchd to pass through for each evaluation step
    
    Input:
    model - trainied classifier model
    device - CPU or CUDA
    trainloader - image sets in dataloader form used for training
    validloader - image sets in dataloader form used for evaluating accuracy
    epochs - no. of times to loop through all images
    save_dir - directory to save checkpoint
    model_spec - model specification for later rebuilding the model
    
    Output:
    Checkpoint file saved in save_dir
    
    '''
    # parameters
    criterion, optimizer = model_spec['criterion'], model_spec['optimizer']
    epochs = epochs
    eval_every = 10
        
    # for timing
    n_img = len(trainloader.dataset.imgs)
    n_batch_per_epoch = round(n_img/trainloader.batch_size)
    n_batch = n_batch_per_epoch * epochs
        
    with active_session():
        # counter reset
        steps = 0 # counter for total no. of batches
        list_tloss, list_vloss, list_step = [], [], []
            
        model.to(device)
            
        print('Training has started.')
            
        for e in range(epochs):
            # counter reset
            running_loss = 0
            estep = 0 # counter for no. of batches per epoch
                
            model.train() # Train mode on

            for images, labels in trainloader:
                start_time = time() # timing
                    
                # Ensure parameters are reset at start
                images, labels = images.to(device), labels.to(device)
                steps += 1
                estep  += 1
                optimizer.zero_grad()

                # Feedforward & Backpropagation
                outputs = model.forward(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Calculation of Loss averaged across every (print_every) iterations
                running_loss += loss.item()
                    
                end_time = time()
                    
                # Evaluation part, with results printed on screen
                if estep % eval_every == 0:
                   
                    time_per_batch = end_time - start_time # timing
 
                    model.eval() 

                    # Evaluation with Validation Set
                    with torch.no_grad():
                        valid_loss, accuracy = validation(model, validloader, criterion, device)
                            
                    eva_time= time()-end_time
                        
                    remaining_batch = n_batch - steps
                    remaining_time = remaining_batch * time_per_batch + eva_time * round(remaining_batch/10)
                    hours, minutes, seconds = int(remaining_time / 3600), int((remaining_time % 3600) / 60), int((remaining_time % 3600) % 60)
                            
                    print(f"Epoch: {e+1}/{epochs}", 
                          f"Training Loss: {running_loss/eval_every :.3f}",
                          f"Valid Loss: {valid_loss :.3f}", 
                          f"Valid Accuracy: {accuracy*100 :.2f}%",
                          f"Remaining Time: {hours}hr {minutes}min {seconds}s")

                    # Recoding each evaluation results
                    list_tloss.append(running_loss/eval_every)
                    list_vloss.append(valid_loss)
                    list_step.append(steps)

                    model.train()
                    running_loss = 0

                # Break the loop id Test mode is ON.
                if test_mode:
                    print('Warning: Testing Mode ON!')
                    break

    print('Training completed!')
        
    # Save nothing if Test mode is On
    if not test_mode:
        # Saving a Check Point
        checkpoint = {'model_spec': model_spec,
                      'state_dict': model.state_dict(),
                      'training loss': list_tloss,
                      'valid loss': list_vloss,
                      'steps': list_step,
                      'optimizer_stat_dict' : optimizer.state_dict()}
        
        checkpoint_dir = save_dir + '/checkpoint_' + model_spec['Model'] + '.pth'
        torch.save(checkpoint, checkpoint_dir)
        print(f'''
        Checkpoint saved.
        Checkpoint Directory: {checkpoint_dir}
        ''')
            
    else: 
        print('Test Mode ON: Checkpoint discarded.')