/**
 * ED_INIT_NETWORK.C - Error Diffusion Network Initialization
 *
 * Unlike standard neural networks, ED networks require setup of:
 *
 * 1. Excitatory/Inhibitory neuron type assignments
 * 2. Constrained weight initialization based on neuron type pairs
 * 3. Architectural constraints for proper ED learning dynamics
 * 4. Multi-layer connectivity rules
 *
 * NEURON INDEXING SYSTEM:
 * Index 0-1:           Bias neurons (threshold inputs)
 * Index 2 to in+1:     Input neurons (doubled for +/- pairs)
 * Index in+2:          First hidden/output neuron
 * Index in+3 to all+1: Remaining hidden neurons
 */

#include "ed_main.h"

/**
 * Initialize ED Neural Network with Biological Constraints
 *
 * This function sets up the network architecture with all the special
 * constraints needed for Error Diffusion learning to work properly.
 * The initialization is much more complex than standard neural networks
 * because ED requires specific neuron type patterns and connection rules.
 *
 * @param init_input    Number of input neurons (already doubled for +/- pairs)
 * @param init_output   Number of output neurons
 * @param init_hidden   Number of neurons in first hidden layer
 * @param init_hidden2  Number of neurons in second hidden layer (0 if unused)
 */
void init_network(int init_input, int init_output, int init_hidden, int init_hidden2) {
    int out_network;  // Output network index (for multiple outputs)
    int c_neuron;     // Current (target) neuron index
    int s_neuron;     // Source neuron index

    /* 1: NETWORK ARCHITECTURE SETUP */

    // Store network dimensions
    size_input = init_input;                       // Input neurons (already doubled)
    size_output = init_output;                     // Output neurons
    size_hidden = init_hidden + init_hidden2;      // Total hidden neurons
    size_hidden2 = init_hidden2;                   // Second layer size
    total_neurons = size_input + 1 + size_hidden;  // +1 for bias neuron

    printf("Initializing ED network: %d inputs, %d hidden (%d+%d), %d outputs\n", size_input, size_hidden, init_hidden, init_hidden2, size_output);

    /* 2: NEURON TYPE ASSIGNMENT (EXCITATORY/INHIBITORY) */
    // Each neuron is assigned as either excitatory (+1) or inhibitory (-1) using an alternating pattern.
    // This pattern determines how learning occurs. 

    // Initialize for each output network (usually just one)
    for (out_network = 0; out_network < size_output; out_network++) {
        // Create alternating excitatory/inhibitory pattern:
        // Index 0: +1 (bias+)     Index 1: -1 (bias-)
        // Index 2: +1 (input1+)   Index 3: -1 (input1-)
        // Index 4: +1 (input2+)   Index 5: -1 (input2-)
        // ...and so on
        for (c_neuron = 0; c_neuron <= total_neurons + 1; c_neuron++) {
            // Formula: ((index + 1) % 2) * 2 - 1 creates +1, -1, +1, -1...
            weights_oscillating[c_neuron] = ((c_neuron + 1) % 2) * 2 - 1;
        }

        // SPECIAL CASE: Output neuron is always excitatory
        // This ensures consistent output behavior regardless of
        // the alternating pattern
        weights_oscillating[size_input + 2] = 1;  // First hidden/output is always +1

        /* 3: WEIGHT INITIALIZATION WITH ED CONSTRAINTS */

        // Initialize weights for all possible connections
        for (c_neuron = size_input + 2; c_neuron <= total_neurons + 1; c_neuron++) {
            for (s_neuron = 0; s_neuron <= total_neurons + 1; s_neuron++) {
                // Bias/Threshold Connections (indices 0-1)
                // These provide bias inputs to neurons
                if (s_neuron < 2) {
                    weights[out_network][c_neuron][s_neuron] = init_range_threshold * random();
                }

                // Regular Connections (indices > 1)
                // Standard neuron-to-neuron connections
                if (s_neuron > 1) {
                    weights[out_network][c_neuron][s_neuron] = init_range_weight * random();
                }

                // Second Hidden Layer Input Blocking
                // Prevents second hidden layer from connecting directly to inputs - This creates proper hierarchical processing
                if (c_neuron > total_neurons + 1 - size_hidden2 && s_neuron < size_input + 2 && s_neuron >= 2) {
                    weights[out_network][c_neuron][s_neuron] = 0;
                }

                // Loop Cutting (config_flags[6])
                // Prevents certain recurrent connections between hidden neurons - This maintains feedforward structure when enabled
                if (config_flags[6] == 1 && c_neuron != s_neuron && c_neuron > size_input + 2 && s_neuron > size_input + 1) {
                    weights[out_network][c_neuron][s_neuron] = 0;
                }

                if (config_flags[6] == 1 && c_neuron > size_input + 1 && s_neuron > size_input + 1 && s_neuron < size_input + 3) {
                    weights[out_network][c_neuron][s_neuron] = 0;
                }

                // Multi-layer Input Blocking (config_flags[7])
                // Controls direct input-to-output connections in multi-layer networks - Forces information to flow through hidden layers
                if (config_flags[7] == 1 && s_neuron >= 2 && s_neuron < size_input + 2 && c_neuron >= size_input + 2 && c_neuron < size_input + 3) {
                    weights[out_network][c_neuron][s_neuron] = 0;
                }

                // Second Hidden Layer Internal Connections
                // Allows connections within the second hidden layer
                if (c_neuron > total_neurons + 1 - size_hidden2 && s_neuron >= size_input + 3) {
                    weights[out_network][c_neuron][s_neuron] = init_range_weight * random();
                }

                // Self-Loop Control (config_flags[3])
                // Determines whether neurons can connect to themselves - Self-loops can create memory effects in the network
                if (c_neuron == s_neuron) {
                    if (config_flags[3] == 1) {
                        weights[out_network][c_neuron][s_neuron] = 0;  // No self-loops
                    } else {
                        weights[out_network][c_neuron][s_neuron] = init_range_weight * random();
                    }
                }

                // Inhibitory Input Control (config_flags[11])
                // Controls whether inhibitory inputs are used - When disabled, removes connections from inhibitory input neurons
                if (config_flags[11] == 0 && s_neuron < size_input + 2 && (s_neuron % 2) == 1) {
                    weights[out_network][c_neuron][s_neuron] = 0;
                }

                // Each weight is multiplied by the product of source and target neuron types (+1 or -1) - This creates the constraint pattern:
                // Excitatory → Excitatory: (+1) × (+1) = +1 (positive weight)
                // Inhibitory → Inhibitory: (-1) × (-1) = +1 (positive weight)
                // Excitatory → Inhibitory: (+1) × (-1) = -1 (negative weight)
                // Inhibitory → Excitatory: (-1) × (+1) = -1 (negative weight)
                weights[out_network][c_neuron][s_neuron] *= weights_oscillating[s_neuron] * weights_oscillating[c_neuron];
            }
        }

        // Initialize bias inputs for this output network
        // Both positive and negative bias neurons get same value
        neuron_input[out_network][0] = bias;  // Positive bias
        neuron_input[out_network][1] = bias;  // Negative bias
    }

    /* 4: RESET LEARNING COUNTERS */
    
    error_count = 0;  // Reset error pattern counter
    error_total = 0;  // Reset total accumulated error
}