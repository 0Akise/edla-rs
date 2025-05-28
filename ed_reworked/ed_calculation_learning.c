/**
 * ED_CALCULATION_LEARNING.C - The "Chemical Diffusion" Core
 *
 * This file implements the Error Diffusion mechanism that gives the algorithm its name.
 * Instead of apply BP gradients through layers, ED broadcasts error signals uniformly to ALL neurons simultaneously,
 * like neurotransmitters diffusing through brain tissue.
 *
 * THE CONCEPT:
 * Traditional Backprop: Different gradient for each layer
 * Layer 3: ∂E/∂w₃ = f₃'(x) × δ₃
 * Layer 2: ∂E/∂w₂ = f₂'(x) × δ₂ × w₃ × f₃'(x) ← Complex chain
 * Layer 1: ∂E/∂w₁ = f₁'(x) × δ₁ × w₂ × f₂'(x) × w₃ × f₃'(x)
 *
 * Error Diffusion: SAME error signal to ALL layers
 * All Layers: Δw = α × input × σ'(x) × SAME_ERROR × neuron_types
 *
 * BIOLOGICAL INSPIRATION:
 * In real brains, neurotransmitters like dopamine create "global" signals that affect many neurons simultaneously,
 * rather than precise point-to-point gradient information. ED mimics this by broadcasting error signals.
 */

#include "ed_main.h"

/**
 * Error Diffusion Signal Generator
 *
 * This function implements the core of Kaneko's method:
 * converting output errors into broadcasted "chemical" signals that diffuse uniformly to all network layers.
 *
 * The process:
 * 1. Calculate prediction error at output layer
 * 2. Split error into excitatory/inhibitory channels based on sign
 * 3. Broadcast SAME error signal to ALL hidden layers
 * 4. Apply amplification for hidden layer learning
 *
 * @param target_pattern_current  Target output values for this pattern
 */
void calculate_learning(double* target_pattern_current) {
    int out_network;  // Output network index (for multiple outputs)
    int s_neuron;     // Source/output neuron index
    int c_neuron;     // Current hidden neuron index

    double error_prediction;  // Raw prediction error (target - actual)
    double error_excitatory;  // Positive error signal (increase output)
    double error_inhibitory;  // Negative error signal (decrease output)

    printf("Starting Error Diffusion calculation...\n");

    // 1: OUTPUT LAYER ERROR CALCULATION
    // For each output neuron, calculate the prediction error and
    // accumulate statistics. This is similar to standard neural networks.
    for (s_neuron = 0; s_neuron <= size_output - 1; s_neuron++) {
        // Calculate basic prediction error:
        // Positive error = need to increase output
        // Negative error = need to decrease output
        error_prediction = target_pattern_current[s_neuron] - neuron_output[s_neuron][size_input + 2];

        // Accumulate total error for convergence monitoring
        error_total += fabs(error_prediction);

        // Count patterns with significant error (threshold = 0.5)
        if (fabs(error_prediction) > 0.5) {
            error_count++;
        }

        printf("Output %d: target=%.3f, actual=%.3f, error=%.3f\n", s_neuron, target_pattern_current[s_neuron], neuron_output[s_neuron][size_input + 2], error_prediction);

        // 2: ERROR CHANNEL SPLITTING
        // Instead of computing gradients, ED splits the error into two simple channels based on the error's SIGN:
        // EXCITATORY CHANNEL [0]: Used when output needs to INCREASE
        // INHIBITORY CHANNEL [1]: Used when output needs to DECREASE
        //
        // This binary splitting mimics how biological neural networks use different neurotransmitter systems for excitation vs inhibition.
        if (error_prediction > 0) {
            // POSITIVE ERROR: Output too low, need to increase it
            // Route error to EXCITATORY channel:
            // - Excitatory channel gets the error magnitude
            // - Inhibitory channel gets zero
            //
            // This will strengthen excitatory pathways and weaken inhibitory pathways to increase output.
            error_delta[s_neuron][size_input + 2][0] = error_prediction;  // Excitatory
            error_delta[s_neuron][size_input + 2][1] = 0;                 // Inhibitory

            printf("  → Excitatory error signal: %.3f\n", error_prediction);

        } else {
            // NEGATIVE ERROR: Output too high, need to decrease it
            // Route error to INHIBITORY channel:
            // - Excitatory channel gets zero
            // - Inhibitory channel gets error magnitude (made positive)
            //
            // This will strengthen inhibitory pathways and weaken excitatory pathways to decrease output.
            error_delta[s_neuron][size_input + 2][0] = 0;                  // Excitatory
            error_delta[s_neuron][size_input + 2][1] = -error_prediction;  // Inhibitory (positive)

            printf("  → Inhibitory error signal: %.3f\n", -error_prediction);
        }

        // Cache the split error signals for broadcasting
        error_excitatory = error_delta[s_neuron][size_input + 2][0];
        error_inhibitory = error_delta[s_neuron][size_input + 2][1];

        // 3: ERROR SIGNAL BROADCASTING (CHEMICAL DIFFUSION)
        // Instead of computing different gradients for each layer (like BP), ED broadcasts the SAME error signal to ALL hidden neurons simultaneously.
        // Think of this like a neurotransmitter being released into the extracellular space: it affects ALL nearby neurons equally, not just specific targeted synapses.
        //
        // CRITICAL INSIGHT: Every hidden neuron receives the SAME error signal.
        // The learning direction is then determined by the neuron type constraints in the weight update , not by layer-specific gradients.
        printf("  Broadcasting error signals to all %d hidden neurons...\n", total_neurons + 1 - (size_input + 3) + 1);

        for (c_neuron = size_input + 3; c_neuron <= total_neurons + 1; c_neuron++) {
            // UNIFORM ERROR BROADCASTING WITH AMPLIFICATION
            // Every hidden neuron gets:
            // 1. The SAME excitatory error signal (amplified)
            // 2. The SAME inhibitory error signal (amplified)
            //
            // The amplification factor allows hidden layers to learn more aggressively than the output layer if needed.
            error_delta[s_neuron][c_neuron][0] = error_excitatory * error_amplification;
            error_delta[s_neuron][c_neuron][1] = error_inhibitory * error_amplification;

            // Debug output for first few neurons
            if (c_neuron <= size_input + 5) {
                printf("    Neuron %d: excitatory=%.3f, inhibitory=%.3f\n", c_neuron, error_delta[s_neuron][c_neuron][0], error_delta[s_neuron][c_neuron][1]);
            }
        }
    }

    // What is fundamentally different from traditional neural network learning:
    // BP APPROACH:
    //     - Compute different gradients for each layer
    //     - Requires chain rule calculations
    //     - Gradients diminish through layers (vanishing gradient problem)
    //     - Sequential, layer-by-layer processing
    //     - Mathematically precise but biologically implausible
    //
    // ED APPROACH:
    //     - Broadcast SAME error to all layers
    //     - No gradient calculations needed
    //     - No vanishing gradient problem
    //     - Parallel processing possible
    //     - Biologically plausible (like neurotransmitter diffusion)
    //
    // INSIGHTS:
    // 1. Same error signal → All layers learn from same "goal"
    // 2. Neuron types → Determine individual learning directions
    // 3. Amplification → Controls hidden layer learning strength
    // 4. Binary channels → Simplifies complex gradient computations
    //
    // This is why ED can:
    // - Solve XOR in ~5 steps (vs 100+ for backprop)
    // - Handle deep networks without vanishing gradients
    // - Process all layers in parallel
    // - Work with minimal parameter tuning

    printf("Error Diffusion complete.\n");
}