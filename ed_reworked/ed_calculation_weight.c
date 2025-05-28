/**
 * ED_CALCULATION_WEIGHT.C - The Heart of Error Diffusion Learning
 *
 * This file implements Error Diffusion weight update rule that replaces BP's gradient-based learning.
 *
 * KEY CONCEPTS:
 * 1. No Gradients: Uses neuron types instead of gradient calculations
 * 2. Simultaneous Updates: All weights updated at once (parallel)
 * 3. Chemical Metaphor: Error "diffuses" like neurotransmitters
 * 4. Directional Learning: +/- neuron types determine update direction
 *
 * ED WEIGHT UPDATE RULE:
 * Δw = α × input × σ'(output) × error_signal × neuron_type_interaction
 *
 * Where:
 * - α (alpha): Learning rate
 * - input: Activation from source neuron
 * - σ'(output): Sigmoid derivative (local gradient)
 * - error_signal: Broadcasted error (not BP'ed)
 * - neuron_type_interaction: Product of source/target neuron types (+1/-1)
 */

#include "ed_main.h"

/**
 * Error Diffusion Weight Update Algorithm
 *
 * This function implements the complete ED learning rule that updates all network weights based on broadcasted error signals.
 * Unlike BP which requires sequential gradient computation, ED updates all weights simultaneously using the "chemical diffusion" principle:
 * 1. Error signals split into excitatory/inhibitory channels
 * 2. Neuron type constraints determining update directions
 * 3. Parallel processing of all weight updates
 */
void calculate_weight() {
    int c_neuron;     // Current (target) neuron index
    int connection;   // Connection (source neuron) index
    int out_network;  // Output network index

    double delta;  // Base weight change magnitude

    printf("Starting Error Diffusion weight updates...\n");

    // PROCESS ALL OUTPUT NETWORKS
    for (out_network = 0; out_network < size_output; out_network++) {
        printf("Updating weights for output network %d\n", out_network);

        // ITERATE THROUGH ALL POSSIBLE CONNECTIONS
        // Unlike BP which processes layer by layer, ED can update all connections simultaneously because the error signal has already been "diffused" to all layers.
        for (c_neuron = size_input + 2; c_neuron <= total_neurons + 1; c_neuron++) {
            for (connection = 0; connection <= total_neurons + 1; connection++) {
                // SKIP DISABLED CONNECTIONS
                // Only process connections that were initialized with non-zero weights (respects architectural constraints)
                if (weights[out_network][c_neuron][connection] != 0) {
                    // 1: CALCULATE BASE WEIGHT CHANGE MAGNITUDE
                    // This part is similar to standard neural networks:
                    // - Learning rate controls update size
                    // - Input activation provides signal strength
                    // - Sigmoid derivative provides local gradient
                    delta = learning_rate * neuron_input[out_network][connection];

                    // Sigmoid derivative: f'(x) = f(x) * (1 - f(x))
                    delta *= fabs(neuron_output[out_network][c_neuron]);
                    delta *= (1 - fabs(neuron_output[out_network][c_neuron]));

                    // 2: APPLY ERROR DIFFUSION LEARNING RULE
                    // Instead of BP'ed gradients, ED uses broadcasted error signals combined with neuron type information to determine update direction.
                    if (config_flags[10] == 1) {
                        // MODE 1: BIDIRECTIONAL ERROR APPLICATION
                        // Uses difference between excitatory and inhibitory error signals. This mode allows the network to "pull" toward targets and "push" away from errors simultaneously.
                        weights[out_network][c_neuron][connection] +=
                            delta * weights_oscillating[c_neuron] *
                            (error_delta[out_network][c_neuron][0] -
                             error_delta[out_network][c_neuron][1]);

                        printf("  Bidirectional update: neuron %d ← %d\n", c_neuron, connection);

                    } else {
                        // MODE 2: SELECTIVE ERROR CHANNEL APPLICATION
                        // Updates are applied based on the TYPE of the source neuron (excitatory vs inhibitory).
                        //
                        // EXCITATORY SOURCE NEURONS (weights_oscillating > 0):
                        // - Use excitatory error channel [0]
                        // - Strengthen connections when output should increase
                        // INHIBITORY SOURCE NEURONS (weights_oscillating < 0):
                        // - Use inhibitory error channel [1]
                        // - Strengthen connections when output should decrease

                        if (weights_oscillating[connection] > 0) {
                            // EXCITATORY SOURCE NEURON PATH
                            // When the source is excitatory (+), we use the excitatory error signal. The weight update direction depends on both source and target types:
                            // Excitatory→Excitatory: (+1) × (+1) = +1 (strengthen)
                            // Excitatory→Inhibitory: (+1) × (-1) = -1 (weaken)
                            weights[out_network][c_neuron][connection] += delta * error_delta[out_network][c_neuron][0] * weights_oscillating[connection] * weights_oscillating[c_neuron];

                            printf("  Excitatory update: %c%d ← %c%d (factor=%.3f)\n", weights_oscillating[c_neuron] > 0 ? '+' : '-', c_neuron, '+', connection, error_delta[out_network][c_neuron][0]);
                        } else {
                            // INHIBITORY SOURCE NEURON PATH
                            // When the source is inhibitory (-), we use the inhibitory error signal. Again, the final direction depends on the target neuron type:
                            // Inhibitory→Inhibitory: (-1) × (-1) = +1 (strengthen)
                            // Inhibitory→Excitatory: (-1) × (+1) = -1 (weaken)
                            weights[out_network][c_neuron][connection] += delta * error_delta[out_network][c_neuron][1] * weights_oscillating[connection] * weights_oscillating[c_neuron];

                            printf("  Inhibitory update: %c%d ← %c%d (factor=%.3f)\n", weights_oscillating[c_neuron] > 0 ? '+' : '-', c_neuron, '-', connection, error_delta[out_network][c_neuron][1]);
                        }
                    }
                    // This learning rule creates four distinct behaviors:
                    // 1. E→E with excitatory error: Strengthen connection
                    // 2. I→I with inhibitory error: Strengthen connection
                    // 3. E→I with excitatory error: Weaken connection
                    // 4. I→E with inhibitory error: Weaken connection
                    //
                    // This pattern ensures that the network learns to:
                    // - Increase output through excitatory pathways
                    // - Decrease output through inhibitory pathways
                    // - All without computing gradients.
                }
            }
        }
    }
    // WEIGHT UPDATE COMPLETE
    // At this point, ALL network weights have been updated simultaneously based on the broadcasted error signal. This is fundamentally different from BP's sequential layer-by-layer updates.
    //
    // Key advantages of this approach:
    // 1. Parallel processing: All updates can happen simultaneously
    // 2. Biological plausibility: Mimics chemical diffusion in brain
    // 3. Computational simplicity: No gradient calculations required
    // 4. Robustness: Less sensitive to vanishing gradients
    // 5. Speed: Potentially faster than BP for deep networks
    printf("Error Diffusion weight updates complete.\n");
}