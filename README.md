**Induction heads in looped 1-layer transformers**

**Summary:** This project investigates whether induction heads can form in weight-tied (looped) transformers with 2 layers which have identical weights. You can also think of these as "1 layer 2 loop" transformers. We use a GPT2-style model architecture. Induction heads are part of a relatively simple 2-layer circuit which aids in bigram completion, that is, if AB appear early in input, and A appears later, induction heads boost B in the logits for A. Induction heads do not form in 1 layer transformers, but do form in 2 layer non-weight-tied transformers. We verify that induction heads do form in 1-layer 2-loop models, so what matters is passing through 2 layers total rather than having different weights in the layers. We also trace the induction circuit in both weight-tied and non-weight-tied models, and find the mechanism to be very similar.

**Results:** We trained a GPT2-style model with tied weights (45.5B parameters, 7B in transformer blocks) and with untied weights (54B parameters, 14B in transformer blocks) as control. In training, both the tied model and the control form induction heads:

<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/18b33555-ac8f-4135-91f8-06760d948748" />
This plot shows mean attention of a particular head towards the "correct" token over the course of the training run. That is, the token the head should attend to if it is an induction head.

We also see the characteristic induction-circuit attention patterns in the tied mode. with input (random tokens, same random tokens).There is a previous-token attention head in Layer 0 (attention on the subdiagonal) and an induction head in Layer 1 (once repeat starts, attention on the shifted subdiagonal).

<img width="2380" height="1224" alt="image" src="https://github.com/user-attachments/assets/2a463c14-fa4f-4afe-83d4-e5a2edfe49b8" />

