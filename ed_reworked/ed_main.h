/**
 * ED_MAIN.H - Error Diffusion Neural Network Learning Method
 *
 * This header defines the data structures and interfaces for Kaneko's Error Diffusion Learning Algorithm (EDLA),
 * a biologically-inspired alternative to BP that uses chemical diffusion instead of gradient-based error propagation.
 *
 * Key Concepts:
 * - Excitatory (+) and Inhibitory (-) neuron types
 * - Error signals broadcast via "chemical diffusion" to all layers
 * - Directional learning based on neuron type combinations
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
// #include "ed_graphic.h"

// Maximum network dimensions
#define MAX_NETWORK 1000  // Maximum neurons per layer
#define MAX_OUTPUT 10     // Maximum output neurons

/* NETWORK ARCHITECTURE PARAMETERS */

// Core network dimensions
extern int size_input;     // Number of input neurons (doubled for +/- pairs)
extern int size_output;    // Number of output neurons
extern int size_hidden;    // Total hidden neurons (size_hidden + size_hidden2)
extern int size_hidden2;   // Second hidden layer size (for multi-layer networks)
extern int total_neurons;  // Total neurons in network (input + bias + hidden)

/* NEURON STATE ARRAYS */
// These 3D arrays organize neuron data by [output_network][neuron_id][...]
// - First dimension: which output neuron's network (usually 0)
// - Second dimension: neuron index in the network
// - Third dimension: additional data (for error_delta: [0]=excitatory, [1]=inhibitory)

// Neuron activation values
extern double neuron_input[MAX_OUTPUT + 1][MAX_NETWORK + 1];   // Input to each neuron
extern double neuron_output[MAX_OUTPUT + 1][MAX_NETWORK + 1];  // Output from each neuron

// Error signals for ED learning - split into excitatory/inhibitory channels
// [neuron][0/1] = excite/inhibit error
extern double error_delta[MAX_OUTPUT + 1][MAX_NETWORK + 1][2];

/* WEIGHT SYSTEM */
// ED method uses constrained weights based on neuron types:
// - Excitatory-to-Excitatory: positive weights
// - Inhibitory-to-Inhibitory: positive weights
// - Excitatory-to-Inhibitory: negative weights
// - Inhibitory-to-Excitatory: negative weights

// Connection weights: [output_network][target_neuron][source_neuron]
extern double weights[MAX_OUTPUT + 1][MAX_NETWORK + 1][MAX_NETWORK + 1];

// Alternating +1/-1 pattern defining neuron types (excitatory/inhibitory)
// Index i: weights_oscillating[i] = +1 (excitatory) or -1 (inhibitory)
extern double weights_oscillating[MAX_NETWORK + 1];

/* LEARNING PARAMETERS */

extern double learning_rate;        // controls weight update magnitude
extern double bias;                 // bias input to all neurons
extern double sigmoid_steepness;    // controls sigmoid function steepness
extern double error_amplification;  // amplifies error signals in hidden layers

/* ERROR TRACKING */

extern double error_total;     // Accumulated error across all patterns
extern double error_residual;  // Target residual error for convergence
extern int error_count;        // Count of patterns with significant error

/* CONFIGURATION FLAGS */
// These flags control various network behaviors and constraints:
// [3] - Self-loop connections (neuron connecting to itself)
// [6] - Inter-layer loop cutting
// [7] - Multi-layer flag
// [10] - Weight decrement mode
// [11] - Input connection handling

extern int config_flags[15];         // Configuration switches
extern double init_range_weight;     // Weight initialization range
extern double init_range_threshold;  // Threshold initialization range

/* TRAINING DATA & CONTROL */

extern int timesteps;                      // Recurrent processing steps per forward pass
extern int pattern_type[MAX_NETWORK + 1];  // Pattern generation types (XOR, parity, etc.)

// Training pattern storage: [pattern_index][input/output_index]
extern double input_pattern_current[MAX_NETWORK + 1];
extern double target_pattern_current[MAX_NETWORK + 1];

/* UTILITY FUNCTIONS */

extern double signage(double x);  // Sign function: returns -1, 0, or 1
extern double random();           // Random number generator [0,1]
extern double sigmoid(double x);  // Sigmoid activation function

/* CORE ED ALGORITHM FUNCTIONS */

extern void init_network(int init_input, int init_output, int init_hidden, int init_hidden2);  // Network initialization
extern void user_input_network_params();                                                       // parameter setup from user input

extern void calculate_output(double* input_pattern_current);                                   // Forward propagation
extern void calculate_learning(double* target_pattern_current);                                // Error diffusion calculation
extern void calculate_weight();                                                                // Weight updates via ED rule
extern void calculate_network(double* input_pattern_current, double* target_pattern_current);  // Complete forward + learning pass

extern void write_output_neuron(int write_mode, double* target_pattern_current);  // Output and monitoring
extern int write_weight_neuron(int loop, int write_mode, int pattern_count);      // Output and monitoring
