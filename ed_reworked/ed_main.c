/**
 * ED_MAIN.C - Error Diffusion Neural Network Main Program
 *
 * This file implements the main training loop for Kaneko's Error Diffusion learning algorithm.
 *
 * Program Flow:
 * 1. Initialize network parameters and architecture
 * 2. Generate or load training patterns
 * 3. Run training epochs until convergence
 * 4. Monitor and visualize learning progress
 */

#include "ed_main.h"

/* GLOBAL TRAINING DATA STORAGE */

// These arrays store all training patterns:
// - input_pattern_global[pattern_id][input_feature]
// - target_pattern_global[pattern_id][output_target]
double input_pattern_global[MAX_NETWORK + 1][MAX_NETWORK + 1];   // All input patterns
double target_pattern_global[MAX_NETWORK + 1][MAX_NETWORK + 1];  // All target outputs

/* GLOBAL VARIABLE DEFINITIONS */
// All the network state variables declared in the header are defined here with their actual memory allocation.

// Network architecture
int size_input;
int size_output;
int size_hidden;
int size_hidden2;
int total_neurons;

// Neuron states (3D arrays for multiple output networks)
double neuron_input[MAX_OUTPUT + 1][MAX_NETWORK + 1];
double neuron_output[MAX_OUTPUT + 1][MAX_NETWORK + 1];
double error_delta[MAX_OUTPUT + 1][MAX_NETWORK + 1][2];  // [neuron][excite/inhibit]

// Weight system
double weights[MAX_OUTPUT + 1][MAX_NETWORK + 1][MAX_NETWORK + 1];  // [output][to][from]
double weights_oscillating[MAX_NETWORK + 1];                       // +1/-1 pattern for neuron types

// Learning parameters
double learning_rate, bias;
double sigmoid_steepness, error_amplification;

// Error tracking
double error_total, error_residual;
int error_count;

// Configuration and initialization
int config_flags[15];
double init_range_weight, init_range_threshold;
double input_pattern_current[MAX_NETWORK + 1], target_pattern_current[MAX_NETWORK + 1];

// Training control variables
int epoch_counter;                  // Current training epoch
int loop_pattern;                   // Current pattern being processed
int timesteps;                      // Recurrent processing steps
int pattern_type[MAX_NETWORK + 1];  // Types of patterns to generate
int max_iterations;                 // Maximum training iterations
int pattern_count;                  // Total number of training patterns
int print_position;                 // Display formatting
int write_mode;                     // Output verbosity level

/* External function declaration */
extern void learning_pattern_generation();

/* USER INPUT FUNCTIONS */
// These functions handle parameter input with default values, allowing users to simply press Enter to use defaults

// Get integer input from user with default fallback
void user_input_int(char* s, int* value, int default) {
    char buffer[128];
    printf("%s", s);

    if (gets(buffer) && *buffer)
        *value = atoi(buffer);
    else
        *value = default;
}

// Get floating-point input from user with default fallback
void user_input_float(char* s, double* value, double default) {
    char buffer[128];
    printf("%s", s);

    if (gets(buffer) && *buffer)
        *value = atof(buffer);
    else
        *value = default;
}

/* UTILITY FUNCTIONS */

// Sign function - returns the mathematical sign of a number Used in ED weight updates to determine learning direction
double signage(double x) {
    return (x > 0.0) ? 1.0 : ((x == 0.0) ? 0.0 : -1.0);
}

// Sigmoid activation function with configurable steepness - The steepness parameter (sigmoid_steepness) controls how sharp the sigmoid transition is.
double sigmoid(double x) {
    return (1 / (1 + exp((double)(-2 * x / sigmoid_steepness))));
}

// Simple random number generator returning value in [0,1] - Uses integer arithmetic for reproducible results
double random() {
    return ((rand() % 10000) / 10000.0);
}

/* TERMINAL CONTROL FUNCTIONS (ANSI Escape Sequences) */

// Position cursor at specific screen coordinates
void locate(int x, int y) {
    putchar(27);  // ESC character
    printf("[%d;%dH", y + 1, x + 1);
}

// Clear screen and move cursor to top-left
void cls() {
    putchar(27);  // ESC character
    printf("[2J");
    locate(0, 0);
}

/* MAIN PROGRAM */

main() {
    init();
    cls();

    // 1: NETWORK ARCHITECTURE SETUP
    printf("=== Error Diffusion Neural Network Learning ===\n");

    // Random seed for reproducible experiments
    int seed;
    user_input_int("Randomized seed? (default=1): ", &seed, 1);

    srand(seed);

    // Network dimensions
    user_input_int("Input neurons? (default=4): ", &size_input, 4);
    user_input_int("Training patterns? (default=16): ", &pattern_count, 16);
    user_input_int("Output neurons? (default=1): ", &size_output, 1);

    // CRITICAL: ED method doubles inputs for excitatory/inhibitory pairs, Each logical input becomes two physical neurons: one + and one -
    size_input *= 2;

    // 2: TRAINING DATA GENERATION
    printf("\nGenerating training patterns...\n");

    // Generate XOR, parity, or custom patterns
    learning_pattern_generation();

    // Hidden layer configuration
    user_input_int("Hidden neurons? (default=8): ", &size_hidden, 8);
    user_input_int("Second hidden layer? (default=0): ", &size_hidden2, 0);

    // Display and debugging options
    user_input_int("Output mode? (0=quiet, 1=verbose, 2=compact, 3=minimal): ", &write_mode, 0);
    user_input_int("Display position offset?: ", &print_position, 0);

    // 3: NETWORK INITIALIZATION
    printf("\nConfiguring ED learning parameters...\n");

    // Set learning rate, sigmoid steepness, etc.
    input_parameters_neuron();

    printf("Initializing network with ED constraints...\n");
    init_network(size_input, size_output, size_hidden, size_hidden2);

    // 4: MAIN TRAINING LOOP
    // Unlike BP, ED updates all weights simultaneously using broadcasted error signals.
    cls();

    epoch_counter = 0;

    // Draw learning curve display area
    box(50, 40, max_iterations + 50, 250);

    // MAIN TRAINING LOOP - runs until convergence or max iterations
    printf("Starting Error Diffusion learning...\n");
    for (;;) {
        epoch_counter++;

        // Optional: clear screen for real-time display
        if (write_mode == 3)
            locate(0, 0);

        // Process all training patterns in this epoch
        for (loop_pattern = 0; loop_pattern <= pattern_count - 1; loop_pattern++) {
            // Position cursor for pattern-specific output
            switch (write_mode) {
                case 0:  // Quiet mode - no pattern display
                    break;
                case 1:  // Verbose mode - one pattern per line
                    locate(0, loop_pattern + print_position);
                    break;
                case 2:  // Compact mode - multiple columns
                    locate(loop_pattern / (30 - print_position) * 20,
                           loop_pattern % (30 - print_position) + print_position);
                    break;
            }

            // CORE ED LEARNING STEP
            //
            // This function call implements the complete ED algorithm:
            // 1. Forward pass through network
            // 2. Calculate error diffusion signals
            // 3. Update all weights simultaneously
            calculate_network(input_pattern_global[loop_pattern], target_pattern_global[loop_pattern]);

            // Display current pattern results
            write_output_neuron(write_mode, target_pattern_global[loop_pattern]);
        }

        // Check for convergence or termination, Updates learning curve display and determines if the network has learned sufficiently
        if (write_weight_neuron(epoch_counter, write_mode, pattern_count)) {
            printf("\nLEARNING COMPLETE\n");
            printf("Converged in %d epochs\n", epoch_counter);

            break;
        }
    }

    printf("\nPress any key to exit...");
    getchar();
    return 0;
}