## RECURRENT NEURAL NETWORKS

Recurrent neural networks are useful to make predictions for data that is sequential, where knowing what has happened before will help us predict what will be next:

![Image of a RNN](./assets/RNN-unrolled.png)
_credit: Chrish Olah_


When we peak inside the network this is what happens:
![Image of a RNN](./assets/LSTM3-SimpleRNN.png)
_credit: Chrish Olah_

The number of iterations will depend of the sequence length we are looking for. If we want a 5 word sequence the network will repeat 5 times. The formulas happening inside the network are:
* _x_t_ is our input in a step _t_. The best way to feed inputs to our network is to one-hot encode them. For this we will create a dictionary with all our inputs mapping it's value to a number.
* _h_t_ is the hidden state at a step _t_. This is the memory of the network, and it is calculated using _x_t_ and _h_t-1_ (The hidden state from the previous iteration). The formula is  **`h_t = activation(W.x_t + U.h_t-1)`** where activation is **_tanh_** or **_ReLU_**, and **_U_** and **_W_** are sets of weigths. The initial hidden state _h_t0_ is usually initialized with al zeros.
* Finally our output or current hidden state _h_t_ is calculated like **`output = V.h_t`**, where _V_ is a set of weights. We could apply softmax to the result to get our probabilies between 1 and 0.

Something important to remember is that the size of our weights _W_, _U_ and _V_ have to be a size that will give us an output the same size as _h_t_. For example if _h_t-1_ is a vector with dimensions *3x1* our weight vectors will need to have a size *3x3* for the result of *3x3.3x1* to be a vector size *3x1*.


## LONG SHORT TERM MEMORY














## RESOURCES
* Chris Olah's tutorial: http://bit.ly/2seO9VI
* Denny Britz tutorial: http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/
