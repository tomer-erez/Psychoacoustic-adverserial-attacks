import parser
import torch
import plot 
import log
import time 
import acoustic_constraints

def save_perturbation():
    pass 

def load_model(args):
    with open...
        model = 
        model.grads.freeze()

    return model

def create_data_loaders():
    pass

def create_loss_fn():
    pass

def create_optimizer():
    pass

def distance_fn(y,y_pred):
    """Functions which measures the distance between the prediction and target"""
    pass

def evaluate(args,model,data_loader):
    model.eval()
    pass 

def test():
    pass
 
 


def train_epoch(args,model,train_data_loader,optimizer,p):
    optimizer = torch.optim.Adam([p], lr=args.lr)
    for epoch in range(args.num_epochs):
        for x, y in train_data_loader:
            optimizer.zero_grad()  # Clear previous gradients
            clean_spectrogram = stft(x)  # Convert audio to spectrogram
            perturbed_spectrogram = clean_spectrogram + p  # Add perturbation
            perturbed_audio = istft(perturbed_spectrogram) # Convert back to audio waveform
            y_pred = model(perturbed_audio) # Forward pass: Feed perturbed audio into the model
            loss = loss_fn(y_pred, y) # get loss
            if args.allow_imperceptibility_penalty: # bonuses to the loss to encourage impercibtibility w.r.t the human speaker
                bonus = acoustic_constraints.get_bonus_for_imperceptibility(perturbed_spectrogram, clean_spectrogram)
                loss+=args.imperceptibility_weight*bonus
            loss.backward()  # Compute gradients
            optimizer.step() # Apply the update to the perturbation

            # Apply constraints on perturbation to ensure imperceptibillity
            p = acoustic_constraints.perturbation_psychoacoustic_constraints(p)

    return p


def train(args,model,train_data_loader,eval_data_loader,optimizer):
    perturbation=acoustic_constraints.apply_JND(torch.init_random())
    train_losses=[]
    eval_losses=[]
    for epoch in args.num_epochs:
        start = time.time()
        perturbation,train_avg_loss = train_epoch(args,model,train_data_loader,optimizer,perturbation)
        eval_loss = evaluate(args,model,eval_data_loader)
        train_losses.append(train_avg_loss)
        eval_losses.append(eval_loss)
        log.log_epoch(args,epoch,train_avg_loss,eval_loss,time.time()-start)
        plot.plot_epochs(args,train_losses,eval_losses)
    return perturbation
    
    


if __name__ == '__main__':
    args = parser.create_arg_parser().parse_args()
    train_data_loader, eval_data_loader,test_data_loader = create_data_loaders(args)
    optimizer=create_optimizer(args)
    loss_fn = create_loss_fn()
    model = load_model(args.model_name)
    perturbation=train(args,model,train_data_loader,eval_data_loader,optimizer)
    test_loss = test(args,model,test_data_loader)
    log.log_test(args,test_loss)   
    save_perturbation(args,model)
