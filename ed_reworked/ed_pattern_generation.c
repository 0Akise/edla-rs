/**
 * ED_LEARNING_INPUT.C - Training Pattern Generation System
 *
 * This file implements a comprehensive pattern generation system for testing the Error Diffusion learning algorithm.
 * Kaneko included multiple classic neural network benchmark problems to demonstrate ED's capabilities.
 *
 * SUPPORTED PATTERN TYPES:
 * 0: Random        - Random input/output pairs
 * 1: Parity        - XOR and N-bit parity checking (ED's specialty!)
 * 2: Mirror        - Symmetry detection problems
 * 3: Manual        - User-defined custom patterns
 * 4: Real Random   - Continuous random values
 * 5: Only One      - Single pattern per output (classification)
 *
 * INPUT GENERATION MODES:
 * - Binary patterns: Generated using bit manipulation for systematic coverage
 * - Random patterns: Generated using random number generator
 */

#include "ed_main.h"

// External references to global pattern storage
extern double input_pattern_global[MAX_NETWORK + 1][MAX_NETWORK + 1];
extern double target_pattern_global[MAX_NETWORK + 1][MAX_NETWORK + 1];

// External variables defined in ed_main.h
extern int size_input, size_output;
extern int size_hidden, size_hidden2;
extern int pattern_type[MAX_NETWORK + 1];
extern int max_iterations;
extern int epoch_counter, loop_pattern;
extern int pattern_count, print_position;
extern int write_mode;

// External utility functions
extern void user_input_int(char* s, int* value, int default);
extern void user_input_float(char* s, double* value, double default);

/**
 * Training Pattern Generation Master Function
 *
 * This function generates all training patterns based on user preferences.
 * It's designed to create the classic neural network benchmark problems
 * that showcase ED's advantages over backpropagation.
 *
 * The function handles both input generation and target calculation,
 * supporting multiple pattern types simultaneously for different outputs.
 */
void learning_pattern_generation() {
    int out_network;  // Output network index
    int c_neuron;     // Current neuron/pattern index
    int s_neuron;     // Source neuron/input index

    int pattern_choice;                       // User's pattern type choice
    int pattern_gen_mode;                     // Input generation mode
    int random_pattern;                       // Random pattern selection
    int pattern_used_flags[MAX_NETWORK + 1];  // Pattern usage tracking

    double init_value;  // Manual input value

    printf("=== ED Training Pattern Generation ===\n");
    printf("Generating patterns to test Error Diffusion learning...\n\n");

    // 1: INPUT GENERATION MODE SELECTION
    user_input_int("Input generation mode (0=binary systematic, 1=random)? (default=0): ", &pattern_gen_mode, 0);
    if (pattern_gen_mode == 0) {
        printf("Using systematic binary patterns - for XOR and parity problems\n");
    } else {
        printf("Using random input patterns - for general function approximation\n");
    }

    // 2: OUTPUT PATTERN TYPE SELECTION - Each output can have a different pattern type, allowing for multi-task learning experiments.
    printf("\nConfiguring output pattern types...\n");
    for (c_neuron = 0; c_neuron <= size_output - 1; c_neuron++) {
        printf("\nOutput %d pattern type:\n", c_neuron);
        printf("  0 = Random targets\n");
        printf("  1 = Parity (XOR for 2 inputs) - ED's strength!\n");
        printf("  2 = Mirror/symmetry detection\n");
        printf("  3 = Manual entry\n");
        printf("  4 = Real-valued random\n");
        printf("  5 = One-hot classification\n");

        user_input_int("Choice? (default=1 for parity): ", &pattern_choice, 1);
        pattern_type[c_neuron] = pattern_choice;

        switch (pattern_choice) {
            case 1:
                printf("  → Parity checking selected\n");
                break;
            case 2:
                printf("  → Mirror detection selected\n");
                break;
            case 0:
                printf("  → Random patterns selected\n");
                break;
            default:
                printf("  → Pattern type %d selected\n", pattern_choice);
                break;
        }
    }

    // 3: PATTERN GENERATION LOOP - Generate all input patterns and their corresponding targets.
    printf("\nGenerating %d training patterns...\n", pattern_count);

    for (c_neuron = 0; c_neuron <= pattern_count - 1; c_neuron++) {
        // INPUT PATTERN GENERATION
        for (s_neuron = 0; s_neuron <= size_input / 2 - 1; s_neuron++) {
            if (pattern_gen_mode == 0) {
                // SYSTEMATIC BINARY PATTERN GENERATION - Uses bit manipulation to create all possible binary input combinations.
                //
                // Pattern mapping:
                // Pattern 0: [0,0,0,0...] → Binary 0000
                // Pattern 1: [1,0,0,0...] → Binary 0001
                // Pattern 2: [0,1,0,0...] → Binary 0010
                // Pattern 3: [1,1,0,0...] → Binary 0011
                // ...and so on

                if ((c_neuron & (1 << s_neuron))) {
                    input_pattern_global[c_neuron][s_neuron] = 1.0;
                } else {
                    input_pattern_global[c_neuron][s_neuron] = 0.0;
                }

            } else {
                // RANDOM INPUT GENERATION - Creates random input patterns for general function approximation problems.
                input_pattern_global[c_neuron][s_neuron] = random();
            }
        }

        // RESET PATTERN USAGE FLAGS
        for (pattern_choice = 0; pattern_choice <= pattern_count - 1; pattern_choice++) {
            pattern_used_flags[pattern_choice] = -1;
        }

        // TARGET PATTERN GENERATION
        for (out_network = 0; out_network <= size_output - 1; out_network++) {
            switch (pattern_type[out_network]) {
                // PATTERN TYPE 0: RANDOM TARGETS - Simple random binary outputs
                case 0:
                    if (random() > 0.5) {
                        target_pattern_global[c_neuron][out_network] = 1.0;
                    } else {
                        target_pattern_global[c_neuron][out_network] = 0.0;
                    }

                    break;

                // PATTERN TYPE 1: PARITY CHECKING (XOR FAMILY)
                // For 2 inputs: This becomes XOR
                // For N inputs: This becomes N-bit parity
                case 1:
                    pattern_choice = 0;  // Count active inputs

                    for (s_neuron = 0; s_neuron <= size_input / 2 - 1; s_neuron++) {  // Count number of 1's in input pattern
                        if (input_pattern_global[c_neuron][s_neuron] > 0.5) {
                            pattern_choice++;
                        }
                    }

                    if (pattern_choice % 2 == 1) {                           // Parity check: odd count = 1, even count = 0
                        target_pattern_global[c_neuron][out_network] = 1.0;  // Odd parity
                    } else {
                        target_pattern_global[c_neuron][out_network] = 0.0;  // Even parity
                    }

                    break;

                // PATTERN TYPE 2: MIRROR/SYMMETRY DETECTION - Checks if input pattern is symmetric (palindromic).
                case 2:
                    pattern_choice = 0;  // Asymmetry flag

                    for (s_neuron = 0; s_neuron <= size_input / 4 - 1; s_neuron++) {  // Check each position against its mirror
                        if (input_pattern_global[c_neuron][s_neuron] !=
                            input_pattern_global[c_neuron][size_input / 2 - 1 - s_neuron]) {
                            pattern_choice = 1;  // Found asymmetry
                        }
                    }

                    if (pattern_choice == 1) {
                        target_pattern_global[c_neuron][out_network] = 0.0;  // Asymmetric
                    } else {
                        target_pattern_global[c_neuron][out_network] = 1.0;  // Symmetric
                    }

                    break;

                // PATTERN TYPE 3: MANUAL INPUT - Allows user to manually specify target values for each input pattern. Useful for custom problems and experimentation.
                case 3:
                    printf("Pattern %d input: ", c_neuron);
                    for (s_neuron = 0; s_neuron <= size_input / 2 - 1; s_neuron++) {
                        printf("%4.2f ", input_pattern_global[c_neuron][s_neuron]);
                    }

                    printf("→ output %d", out_network);
                    putchar('\n');

                    user_input_float("Target value? ", &init_value, 0.0);
                    target_pattern_global[c_neuron][out_network] = init_value;

                    break;

                // PATTERN TYPE 4: REAL-VALUED RANDOM - Continuous random targets for regression
                case 4:
                    target_pattern_global[c_neuron][out_network] = random();

                    break;

                // PATTERN TYPE 5: ONE-HOT CLASSIFICATION - Creates a classification dataset where each pattern belongs to exactly one class.
                case 5:
                    // Initialize all patterns to 0 for this output
                    for (s_neuron = 0; s_neuron <= pattern_count - 1; s_neuron++) {
                        target_pattern_global[s_neuron][out_network] = 0.0;
                    }

                    // Select one random pattern to be positive
                    do {
                        random_pattern = (int)(random() * pattern_count);
                    } while (!(pattern_used_flags[random_pattern] == -1));

                    pattern_used_flags[random_pattern] = 1;
                    target_pattern_global[random_pattern][out_network] = 1.0;

                    break;
            }
        }
    }

    // PATTERN GENERATION COMPLETE
    printf("\nPattern generation complete.\n");
    printf("Generated %d patterns for learning\n", pattern_count);

    // Display sample patterns for verification
    printf("\nSample patterns (first 4):\n");
    for (c_neuron = 0; c_neuron < 4 && c_neuron < pattern_count; c_neuron++) {
        printf("Pattern %d: [", c_neuron);

        for (s_neuron = 0; s_neuron <= size_input / 2 - 1; s_neuron++) {
            printf("%.0f", input_pattern_global[c_neuron][s_neuron]);

            if (s_neuron < size_input / 2 - 1) {
                printf(",");
            }
        }

        printf("] → ");

        for (out_network = 0; out_network < size_output; out_network++) {
            printf("%.0f", target_pattern_global[c_neuron][out_network]);

            if (out_network < size_output - 1) {
                printf(",");
            }
        }

        printf("\n");
    }
}