**Induction heads in looped 1-layer transformers**

**Summary:** This project investigates whether induction heads can form in weight-tied (looped) transformers with 2 layers which have identical weights. You can also think of these as "1 layer 2 loop" transformers. We use a GPT2-style model architecture. Induction heads are part of a relatively simple 2-layer circuit which aids in bigram completion, that is, if AB appear early in input, and A appears later, induction heads boost B in the logits for A. Induction heads do not form in 1 layer transformers, but do form in 2 layer non-weight-tied transformers. We verify that induction heads do form in 1-layer 2-loop models, so what matters is passing through 2 layers total rather than having different weights in the layers. We also trace the induction circuit in both weight-tied and non-weight-tied models, and find the mechanism to be very similar.

**Results:** Model architecture throughout was GPT2-style (tied embedding and unembedding, learned absolute positional embedding, 2 transformer blocks). We trained a model with tied weights (45.5B parameters, 7B in transformer blocks) and with untied weights (54B parameters, 14B in transformer blocks) as control. In training, both the tied model and the control form induction heads:

**Insert chart here**

These charts show mean attention of a particular head towards the "correct" token over the course of the training run. That is, the token the head should attend to if it is an induction head.

We examined attention patterns for the models, using sequences of the form (25 random tokens, the same 25 random tokens). We see that in the tied-weight model, there is a previous-token head in Layer 0, and an induction head in Layer 1 (if AB appeared before, at later appearances of A, the head attends to the previous appearance of B). This is analogous to the non-tied situation.

**Insert figure here**