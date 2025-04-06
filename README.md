# Psychoacoustically-Informed-Adversarial-Attacks-on-Speech-Recognition-Systems

adverserial attacks on speech recognition models, 

focused on perturbations with low imperceptibility likelihood by the human ear.

studying how different perturbation norms/sizes can 

confuse the model, with the goal of finding attacks that can be hidden from the human ear.

# requirements:

use the conda env installation file attached.

``conda env create -f cs236207.yaml``

# run:
```
python main.py 
```

# dataset:


Common voice: the script will download it for you, you might need to provide a login token to huggingface if requested the first time running it




# hyper params:

play around with the size of the attacks via the cli arguments, also specify the norm type

you can add --small_data for testing 


