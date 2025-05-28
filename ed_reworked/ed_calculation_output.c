/**
 * ED_CALCULATION_OUTPUT.C - Forward Pass and Network Orchestration
 *
 * This file implements the forward propagation  of Error Diffusion learning.
 *
 * Unlike standard neural networks, ED networks have special characteristics:
 * 1. Input Doubling: Each logical input feeds both excitatory and inhibitory neurons
 * 2. Recurrent Processing: Multiple timesteps allow feedback and stabilization
 * 3. Simultaneous Processing: All neurons can be computed in parallel
 * 4. No Hidden Gradients: Forward pass doesn't need to store gradients for backprop
 *
 * NETWORK COMPUTATION FLOW:
 * 1. Input Distribution → Both +/- neurons get same input
 * 2. Recurrent Timesteps → Network stabilizes over multiple steps
 * 3. Sigmoid Activation → Standard nonlinear activation
 * 4. Output Generation → Final activations ready for learning
 */

#include "ed_main.h"

/**
 * This function coordinates the complete ED learning cycle:
 * 1. Forward pass (calculate_output)
 * 2. Error diffusion calculation (calculate_learning)
 * 3. Weight updates (calculate_weight)
 *
 * Unlike backpropagation where these steps are sequential and interdependent,
 * ED can perform these operations more independently due to its broadcasting nature.
 *
 * @param input_pattern_current   Input pattern for this training example
 * @param target_pattern_current  Target pattern for this training example
 */
void calculate_network(double* input_pattern_current, double* target_pattern_current) {
    // 1: Forward propagation through the network
    calculate_output(input_pattern_current);

    // 2: Calculate error diffusion signals (like "chemical gradients")
    calculate_learning(target_pattern_current);

    // 3: Update all weights simultaneously using ED rule
    calculate_weight();

    // Unlike BP, all weight updates happen simultaneously based on the broadcasted error signal,
    // this mimics how neurotransmitters might affect multiple synapses at once in biological neural networks.
}

/**
 * Forward Pass Through ED Network
 *
 * This function propagates input signals forward through the network using the special ED architecture.
 *
 * The key differences from standard networks:
 * - Input doubling for excitatory/inhibitory pairs
 * - Recurrent processing over multiple timesteps
 * - Parallel computation of all layers
 *
 * @param input_pattern_current  Array of input values for this pattern
 */
void calculate_output(double* input_pattern_current) {
    int out_network;  // Output network index (for multiple outputs)
    int c_neuron;     // Current neuron being processed
    int connection;   // Connection index for weight summation
    int timestep;     // Current recurrent processing timestep

    double weighted_sum;  // Accumulated weighted input to a neuron

    // PROCESS EACH OUTPUT NETWORK
    // Usually just one, but ED supports multiple parallel outputs.
    for (out_network = 0; out_network < size_output; out_network++) {
        //  1: INPUT DISTRIBUTION TO EXCITATORY/INHIBITORY PAIRS
        // Each logical input value gets distributed to BOTH an excitatory neuron and an inhibitory neuron.
        // This creates the neural substrate needed for directional learning.
        //
        // Input mapping:
        // input_pattern_current[0] → neuron_input[network][2] and [3] (+ and -)
        // input_pattern_current[1] → neuron_input[network][4] and [5] (+ and -)
        // input_pattern_current[2] → neuron_input[network][6] and [7] (+ and -)
        // ...and so on
        for (c_neuron = 2; c_neuron <= size_input + 1; c_neuron++) {
            // Formula: (c_neuron / 2 - 1), maps paired neurons to same input.
            // c_neuron = 2, 3 → input[0]
            // c_neuron = 4, 5 → input[1], etc.
            neuron_input[out_network][c_neuron] = input_pattern_current[(int)(c_neuron / 2) - 1];
        }

        printf("Input distributed to %d excitatory/inhibitory pairs\n", size_input / 2);

        //  2: HIDDEN LAYER INITIALIZATION
        // If loop cutting is enabled, initialize hidden neurons to zero.
        // This ensures clean processing without residual activations from previous patterns.
        if (config_flags[6]) {  // Loop cutting flag (config_flags[6])
            for (c_neuron = size_input + 2; c_neuron <= total_neurons + 1; c_neuron++) {
                neuron_input[out_network][c_neuron] = 0;
            }
            printf("Hidden layer initialized to zero (loop cutting enabled)\n");
        }

        //  3: RECURRENT PROCESSING TIMESTEPS
        // Instead of single-pass computation, the network processes over multiple timesteps, allowing:
        // - Feedback effects between layers
        // - Network stabilization
        // - Dynamic settling behavior
        // - More realistic temporal processing
        // Each timestep allows the network to "think" and refine its internal representations before producing final outputs.
        printf("Beginning recurrent processing over %d timesteps...\n", timesteps);

        for (timestep = 1; timestep <= timesteps; timestep++) {
            printf("Timestep %d/%d: Computing hidden/output layer activations\n", timestep, timesteps);

            // STEP 3A: COMPUTE ACTIVATIONS FOR ALL HIDDEN/OUTPUT NEURONS
            // For each neuron beyond the input layer, compute its activation by summing all weighted inputs and applying the sigmoid function.
            for (c_neuron = size_input + 2; c_neuron <= total_neurons + 1; c_neuron++) {
                weighted_sum = 0;

                // Sum all weighted connections to this neuron
                for (connection = 0; connection <= total_neurons + 1; connection++) {
                    weighted_sum += weights[out_network][c_neuron][connection] * neuron_input[out_network][connection];
                }

                // Apply sigmoid activation function
                neuron_output[out_network][c_neuron] = sigmoid(weighted_sum);
            }

            // STEP 3B: FEEDBACK FOR NEXT TIMESTEP
            // Copy current outputs back to inputs for next timestep.
            // This creates the recurrent processing that allows the network to iterate toward a stable solution.
            // This feedback mechanism is crucial for complex problems where single-pass processing isn't sufficient.
            for (c_neuron = size_input + 2; c_neuron <= total_neurons + 1; c_neuron++) {
                neuron_input[out_network][c_neuron] = neuron_output[out_network][c_neuron];
            }
        }

        printf("Final output: %.6f\n", neuron_output[out_network][size_input + 2]);
    }

    // FORWARD PASS COMPLETE
    // At this point:
    // - All neuron activations have been computed
    // - Network has processed through recurrent timesteps
    // - Outputs are ready for error calculation
    // - No gradients stored (unlike backpropagation!)
    // Unliked BP, we don't need to track computational graphs or store intermediate gradients.
}