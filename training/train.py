import parser
import torch
import plot 
import log
import time 
import os
import sys
import acoustic_constraints
import sound_processing
def save_perturbation():
    pass 

def load_model(args):
    torch.load(args.path_to_model) 

    return model

def early_stopping(all_losses,early_stopping_patience=6):
    if len(all_losses) >= early_stopping_patience + 1:
        last_losses = all_losses[-early_stopping_patience:]  # Last `early_stopping` losses
        best_loss_before_the_patience = min(all_losses[:-early_stopping_patience])  # Best loss before the last `early_stopping` iterations
        # Check if all recent losses are worse than the best loss seen so far
        if all(loss >= best_loss_before_the_patience for loss in last_losses):
            log.info(f"recent losses are {all_losses[-(early_stopping_patience+1):]}, meaning no improvement in the recent {early_stopping_patience} epochs, stopping training and exitting the script!")
            time.sleep(10)#give time to logger before crashing?
            return True
        else:
            return False
    else:
        log.info('not enough data to decide early stopping')
        return False

def create_data_loaders():
    pass

def create_rirs(args):
    """
    Creates a lsit of rirs, later each preprocess call will randomly choose 1 and apply it
    
    """
    Use pre-recorded RIRs from datasets (like ECHOThief or AIR datasets) for realism.
    pass

def create_loss_fn(args):
    pass

def create_optimizer():
    pass

def distance_fn(y,y_pred):
    """Functions which measures the distance between the prediction and target"""
    pass

def evaluate(args,model,p,data_loader,loss_fn,rec_len,rirs):
    model.eval()
    total_losses=0.
    with torch.no_grad():
        for iter_num,x, y in data_loader:
            perturbed_audio, _,_ = sound_processing.preprocess_sound(args,x,p,rirs)
            y_pred = model(perturbed_audio) # Forward pass: Feed perturbed audio into the model
            loss = loss_fn(y_pred, y) # get loss
            total_losses+=loss.item()
            avg_loss = total_losses/(iter_num+1)
            if iter_num >= rec_len-1:
                break
    return avg_loss
def test():
    pass
 
 



def train_epoch(args,model,data_loader,optimizer,p,loss_fn,rec_len,rirs):
    total_losses=0.0
    for iter_num, x, y in data_loader:
        optimizer.zero_grad()  # Clear previous gradients
        #generate the combined sounds of x and p, acounting for rirs, mic filters
        perturbed_audio, perturbed_spectrogram,clean_spectrogram = sound_processing.preprocess_sound(args,x,p,rirs)
        y_pred = model(perturbed_audio)#Model forward pass with perturbed audio
        loss = loss_fn(y_pred, y)# Calculate loss and apply bonus for imperceptibility
        if args.allow_imperceptibility_penalty:# bonus for x,p alignment of sounds, bonus for hiding of the p
            loss += args.imperceptibility_weight * acoustic_constraints.compute_imperceptibility_penalty(perturbed_spectrogram, clean_spectrogram)
        loss.backward() # backprop
        optimizer.step() # step
        p = acoustic_constraints.perturbation_psychoacoustic_constraints(p)#: Apply psychoacoustic constraints to the perturbation, these are the "Lp norm constraints"

        total_losses += loss
        avg_loss = total_losses / (iter_num + 1)
        if iter_num % args.report_interval == 0:
            log.report_iter(args, iter_num, loss.item(), avg_loss.item())
        if iter_num >= rec_len - 1:
            break

    return p, avg_loss


def train(args,p,model,train_data_loader,eval_data_loader,optimizer,loss_fn,rec_len,rirs):
    optimizer = torch.optim.Adam([p], lr=args.lr)
    train_losses=[]
    eval_losses=[]
    for epoch in args.num_epochs:
        start = time.time()
        p,train_avg_loss = train_epoch(args,model,train_data_loader,optimizer,p,loss_fn,rec_len,rirs)
        eval_loss = evaluate(args,model,p,eval_data_loader,loss_fn,rec_len,rirs)
        train_losses.append(train_avg_loss)
        eval_losses.append(eval_loss)
        log.log_epoch(args,epoch,train_avg_loss,eval_loss,time.time()-start)
        plot.plot_epochs(args,train_losses,eval_losses)
        if early_stopping(eval_losses,args.early_stopping):
            sys.exit(0)
    return p
    
    


if __name__ == '__main__':
    args = parser.create_arg_parser().parse_args()
    log.welcome()
    train_data_loader, eval_data_loader,test_data_loader,rec_len = create_data_loaders(args)
    optimizer = create_optimizer(args)
    loss_fn = create_loss_fn(args)
    rirs = create_rirs(args)
    model = load_model(args.model_name)
    model.grads.freeze()
    p=acoustic_constraints.apply_JND(torch.init_random())
    p=train(args,p,model,train_data_loader,eval_data_loader,optimizer,loss_fn,rec_len,rirs)
    test_loss = evaluate(args,model,p,test_data_loader,loss_fn,rec_len,rirs)
    log.log_test(args,test_loss)   
    save_perturbation(args,model)
    log.summerize()