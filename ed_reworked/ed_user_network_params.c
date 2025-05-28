/**
 * ED_USER_NETWORK_PARAMS.C - Network Parameter Configuration Interface
 *
 * This file provides a interface for configuring all the critical parameters that control Error Diffusion learning behavior.
 *
 * PARAMETER CATEGORIES:
 *     LEARNING DYNAMICS:
 *     - Learning rate (α): Controls weight update magnitude
 *     - Error amplification: Scales hidden layer error signals
 *     - Sigmoid steepness: Controls activation function sensitivity
 *
 *     NETWORK ARCHITECTURE:
 *     - Timesteps: Recurrent processing iterations
 *     - Bias: Global bias input to all neurons
 *     - Connection flags: Control network topology
 *
 *     INITIALIZATION:
 *     - Weight ranges: Initial connection strength limits
 *     - Threshold ranges: Initial bias/threshold limits
 */

#include "ed_main.h"

// External utility functions for user input
extern void user_input_int(char* s, int* value, int default);
extern void user_input_float(char* s, double* value, double default);

/**
 * Network Parameter Configuration Interface
 *
 * This function presents a comprehensive interface for setting all ED network parameters.
 * The defaults are carefully chosen based on Kaneko's research and work well for most problems without tuning.
 */
void user_input_network_params() {
    printf("=== Error Diffusion Parameter Configuration ===\n");

    // RECURRENT PROCESSING CONTROL
    printf("RECURRENT PROCESSING:\n");
    user_input_int("Timesteps (recurrent iterations)? (default=2): ", &timesteps, 2);

    printf("  → Using %d timesteps for network stabilization\n", timesteps);
    if (timesteps == 1) {
        printf("     Single timestep = pure feedforward processing\n");
    } else {
        printf("     Multiple timesteps allow recurrent dynamics and better convergence\n");
    }

    // WEIGHT INITIALIZATION PARAMETERS
    printf("\nWEIGHT INITIALIZATION:\n");
    user_input_float("Initial weight range? (default=1.0): ", &init_range_weight, 1.0);
    user_input_float("Initial threshold range? (default=1.0): ", &init_range_threshold, 1.0);

    printf("  → Weights initialized in [0, %.1f] range\n", init_range_weight);
    printf("  → Thresholds initialized in [0, %.1f] range\n", init_range_threshold);
    printf("     Note: Final weights will be scaled by neuron type constraints\n");

    // NETWORK TOPOLOGY FLAGS - These flags control the architectural constraints that make ED learning possible.
    printf("\nNETWORK TOPOLOGY FLAGS:\n");
    user_input_int("Multi-layer flag (force hierarchical processing)? (default=1): ", &config_flags[7], 1);
    printf("  → Multi-layer processing: %s\n", config_flags[7] ? "ENABLED - prevents input shortcuts" : "DISABLED");

    user_input_int("Weight decrement mode (bidirectional updates)? (default=0): ", &config_flags[10], 0);
    printf("  → Weight decrement mode: %s\n", config_flags[10] ? "ENABLED - uses both excitatory and inhibitory errors" : "DISABLED - uses selective error channels");

    user_input_int("Loop cutting (prevent recurrent connections)? (default=1): ", &config_flags[6], 1);
    printf("  → Recurrent loop cutting: %s\n", config_flags[6] ? "ENABLED - more feedforward structure" : "DISABLED - allows more recurrent connections");

    user_input_int("Self-loop cutting (prevent self-connections)? (default=1): ", &config_flags[3], 1);
    printf("  → Self-loop cutting: %s\n", config_flags[3] ? "ENABLED - no neuron self-connections" : "DISABLED - allows neuron memory effects");

    user_input_int("Inhibitory input connections? (default=1): ", &config_flags[11], 1);
    printf("  → Inhibitory inputs: %s\n", config_flags[11] ? "ENABLED - uses both +/- input neurons" : "DISABLED - uses only excitatory inputs");

    // ACTIVATION FUNCTION PARAMETERS
    printf("\nACTIVATION FUNCTION:\n");
    user_input_float("Sigmoid steepness? (default=0.4): ", &sigmoid_steepness, 0.4);

    // ERROR DIFFUSION PARAMETERS
    printf("\nERROR DIFFUSION CONTROL:\n");
    user_input_float("Error amplification for hidden layers? (default=1.0): ", &error_amplification, 1.0);

    // LEARNING DYNAMICS
    printf("\nLEARNING DYNAMICS:\n");
    user_input_float("Learning rate? (default=0.8): ", &learning_rate, 0.8);
    user_input_float("Bias input value? (default=0.8): ", &bias, 0.8);

    // CONVERGENCE CRITERIA
    printf("\nCONVERGENCE CONTROL:\n");
    user_input_float("Residual error threshold? (default=0.0): ", &error_residual, 0.0);

    // PARAMETER SUMMARY AND RECOMMENDATIONS
    printf("\n=== PARAMETER CONFIGURATION COMPLETE ===\n");
    printf("ED network configured with the following key settings:\n");
    printf("    - Learning rate: %.2f (weight update strength)\n", learning_rate);
    printf("    - Sigmoid steepness: %.2f (activation sensitivity)\n", sigmoid_steepness);
    printf("    - Error amplification: %.2f (hidden layer learning)\n", error_amplification);
    printf("    - Timesteps: %d (recurrent processing)\n", timesteps);
    printf("    - Bias: %.2f (global neuron bias)\n", bias);

    printf("\nNetwork topology:\n");
    printf("     - Self-loops: %s\n", config_flags[3] ? "OFF" : "ON");
    printf("     - Loop cutting: %s\n", config_flags[6] ? "ON" : "OFF");
    printf("     - Multi-layer: %s\n", config_flags[7] ? "ON" : "OFF");
    printf("     - Inhibitory inputs: %s\n", config_flags[11] ? "ON" : "OFF");
}