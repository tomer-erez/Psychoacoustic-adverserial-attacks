# Psychoacoustically-Informed-Adversarial-Attacks-on-Speech-Recognition-Systems

universal un-targeted adverserial attacks on speech recognition models, 

focused on perturbations with low imperceptibility likelihood by the human ear.

studying how different perturbation norms/sizes can 

confuse the model, with the goal of finding attacks that can be hidden from the human ear.

# requirements:

use the conda env installation file attached.

``conda env create -f cs236207.yaml``

run with downloading the dataset (omit the flag else)

```
python main.py --download_ds
```

# dataset:


Common voice: you can either download it in script via the flag or get it from:

https://commonvoice.mozilla.org/en/datasets

or the libreespeech (preferably not because wav2vec2 is trained on it, hence the perturbation needs to train on something else)

https://www.openslr.org/12






# hyper params:

play around with the size of the attacks via the cli arguments, also specify the norm type



