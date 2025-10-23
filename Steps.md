## step 1: Input Encoding(TTFS)

input of network should be like this => t= t_max(1-x)

## X_N:

1. We stored maximum activation of each layer of ANN in X_N => These will act as timing windows or thresholds in SNN conversion.
2. High values indicate large dynamic range.
3. Zeros in X_n leads to a dead layer
4. We should use these maxium values but we can change them for better performance. These values prohibit from vanishing, so we should check in which layers vanishing will occure?
   - How should I check Vanishing?
   - How should we utilize X_n values from solving vanishing?
   - What range does my SNN implementation expoet for threshold or time windows?
     - Try raw values first, but keep an eye out for instability, vanishign gradients or inactive neurons.
       - Instability: Erratic loss, wildly fluctuating membrane potentials.
       - Vanishing gradients: Output stays stuck, network wont's train.
       - inactive neurons: Many layers with zero spiking output.

### Steps:

1. Run a forward pass with a dummy input of shape (1,32,32,3) and check which layers are stucked at zeros?
2. Check that are outpu neurons spiking? Is loss changing reasonably?
3. Bring input to histogram

## Concepts

1. t_min : earliest time to spike a neuron
   - we want to mimic from ANN to train SNN, meaning that each layer of SNN should perform same as ANN's one.
   - it should not be 0, hindering the first layer instantly
2. t_max : latest time that a neuron can spike
   - t_max = number_of_layers - a_small_constant = 16 - 1.5 = 24
     - a_small_constant= 1.5

spike_time = t_max - (value \* (t_max - t_min))

t_min = 1

t_max = 25

## Check SNN Model:

##
