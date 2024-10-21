# Generative_replay
Generative_replay for simulating infant language acquisition

Here's the procedure to go through in order to train the synthetic data-augmented LM

    1. Prepare your data.
    2. Train your LM model (or download a checkpoint adn dict.txt).
    3. Save chunk ppl to a datastore.
    4. Get the vector database from the training set
    5. Build the faisee index (prefix-only and prefix+generation)
    
# The generative_replay model is divided into 2 parts
1. word LM: use standard bpe tokenizer
  tested on BLIMP 
2. char LM
  tested on Machine-CDI, see [https://github.com/Jing-L97/Lexical-benchmark]
