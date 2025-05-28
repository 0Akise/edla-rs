/**
 * ED_WRITE.C - Display, Monitoring, and Learning Curve Visualization
 *
 * This file implements the output display system for monitoring ED learning progress.
 * It provides multiple visualization modes and tracks learning metrics to demonstrate ED's convergence.
 *
 * KEY FEATURES:
 *     DISPLAY MODES:
 *     0: Silent - No pattern output (fastest)
 *     1: Verbose - Full input/output/hidden neuron display
 *     2: Compact - Condensed digit display for large pattern sets
 *     3: Minimal - Only output values (real-time monitoring)
 *
 *     MONITORING:
 *     - Real-time learning curve visualization
 *     - Error counting and convergence detection
 *     - Weight matrix display for analysis
 *     - Training progress indicators
 */

#include "ed_main.h"

// External terminal control function
extern void locate(int, int);

/**
 * Pattern Output Display Function
 *
 * This function displays the current pattern's input, output, and hidden neuron activations in various formats.
 * The display mode determines the level of detail shown during training.
 *
 * @param write_mode              Display verbosity level (0-3)
 * @param target_pattern_current  Target output for comparison
 */
void write_output_neuron(int write_mode, double* target_pattern_current) {
    int c_neuron;  // Current neuron index for iteration

    switch (write_mode) {
        // MODE 0: SILENT OPERATION - No output display for maximum training speed
        case 0:
            break;

        // MODE 1: VERBOSE DISPLAY - Shows complete information about current pattern.
        case 1:
            // Display input values (only from excitatory neurons)
            // Remember: inputs are doubled, so we skip inhibitory pairs
            printf("inputs: ");
            for (c_neuron = 1; c_neuron <= size_input / 2; c_neuron++) {
                printf("%4.2f ", neuron_input[0][c_neuron * 2]);  // Even indices = excitatory
            }

            // Display actual output vs target
            printf("-> ");
            printf("%7.5f, %4.2f ", neuron_output[0][size_input + 2], target_pattern_current[0]);

            // Display first few hidden neuron activations
            printf("hidden: ");
            for (c_neuron = size_input + 3; c_neuron <= size_input + 6; c_neuron++) {
                if (c_neuron <= total_neurons + 1) {
                    printf("%7.4f ", neuron_output[0][c_neuron]);
                }
            }
            break;

        // MODE 2: COMPACT DIGIT DISPLAY - Condenses neuron activations to single digits (0-9) based on activation strength.
        case 2:
            // Display target as single digit
            printf("%1d", (int)(target_pattern_current[0] * 9.999));
            printf(": ");

            // Display all neuron outputs as single digits
            for (c_neuron = size_input + 2; c_neuron <= total_neurons + 1; c_neuron++) {
                printf("%1d", (int)(fabs(neuron_output[0][c_neuron]) * 9.999));  // Convert activation [0,1] to digit [0,9]

                if (c_neuron == size_input + 2) {
                    printf(" ");  // Add space after output neuron for readability
                }
            }
            break;

        // MODE 3: MINIMAL REAL-TIME DISPLAY - Shows only target and output values for real-time monitoring of learning progress.
        case 3:
            // Display target as single digit
            printf("%1d", (int)(target_pattern_current[0] * 9.999));
            printf(":");

            // Display only output neuron, then break
            for (c_neuron = size_input + 2; c_neuron <= total_neurons + 1; c_neuron++) {
                printf("%1d", (int)(fabs(neuron_output[0][c_neuron]) * 9.999));

                if (c_neuron == size_input + 2) {
                    printf(" ");
                    break;  // Only show output, not hidden neurons
                }
            }
            break;
    }

    if (write_mode < 3) {
        putchar('\n');  // Add newline for modes 0-2 (mode 3 updates in place)
    }
}

/**
 * Learning Progress Monitor and Convergence Detector
 *
 * This function handles epoch-level monitoring, weight display, learning curve visualization, and convergence detection.
 * It's the central hub for tracking ED learning progress.
 *
 * @param loop           Current epoch number
 * @param write_mode     Display verbosity level
 * @param pattern_count  Total number of training patterns
 * @return               1 if training should stop, 0 to continue
 */
int write_weight_neuron(int loop, int write_mode, int pattern_count) {
    int c_neuron;      // Current neuron index
    int s_neuron;      // Source neuron index
    int pos_x, pos_y;  // Graphics coordinates for learning curve

    // WEIGHT MATRIX DISPLAY (Mode 0 only) - Shows the complete weight matrix for detailed analysis.
    if (write_mode == 0) {
        printf("\n=== WEIGHT MATRIX ANALYSIS ===\n");
        printf("Connection types: th+   th-   in1+  in1-  in2+  in2-  ...\n");
        printf("(th = threshold/bias, in = input, + = excitatory, - = inhibitory)\n\n");

        // Display weight matrix row by row
        for (c_neuron = size_input + 2; c_neuron <= total_neurons + 1; c_neuron++) {
            printf("Neuron %2d: ", c_neuron);
            for (s_neuron = 0; s_neuron <= total_neurons + 1; s_neuron++) {
                printf("%6.2f ", weights[0][c_neuron][s_neuron]);
            }

            putchar('\n');
        }
    }

    // LEARNING PROGRESS DISPLAY
    locate(0, 29);  // Position cursor for status display
    printf("Error patterns: %3d, Epoch: %d", error_count, loop);

    // Add learning status indicator
    if (error_count == 0) {
        printf(" ✓ PERFECT!");
    } else if (error_count <= pattern_count * 0.1) {
        printf(" ✓ Excellent");
    } else if (error_count <= pattern_count * 0.3) {
        printf(" → Good");
    } else {
        printf(" → Learning...");
    }

    printf("                \n");  // Clear any previous text
    fflush(stdout);

    // LEARNING CURVE VISUALIZATION- Draw real-time learning curve showing error reduction over epochs.
    pos_y = (int)(250 - 200 * error_total / pattern_count / size_output);  // Calculate curve position
    pos_x = loop + 50;

    line(pos_x, 250, pos_x, pos_y);  // Draw point on learning curve (using custom graphics library)
    flush();

    // CONVERGENCE DETECTION AND TERMINATION - Check if learning has converged or reached maximum iterations.
    if (loop > 10000) {
        // MAXIMUM ITERATIONS REACHED
        locate(0, 0);
        printf("\n=== TRAINING TERMINATED ===\n");
        printf("Maximum iterations (%d) reached\n", loop);
        printf("Final error: %.4f\n", error_total);
        printf("Error patterns: %d/%d (%.1f%%)\n", error_count, pattern_count, 100.0 * error_count / pattern_count);

        if (error_count <= pattern_count * 0.1) {
            printf("Result: GOOD - Network has learned well.\n");
        } else {
            printf("Result: BAD - Requires training or parameter adjustment.\n");
        }

        return 1;  // Terminate training
    } else if (error_total < 0.1) {
        // CONVERGENCE ACHIEVED
        locate(0, 0);
        printf("\n=== ED CONVERGENCE ACHIEVED ===\n");
        printf("Converged in %d epochs\n", loop);
        printf("Final total error: %.6f (threshold: 0.1)\n", error_total);
        printf("Error patterns: %d/%d\n", error_count, pattern_count);

        printf("\nED LEARNING SUCCESS METRICS:\n");
        printf("- Convergence speed: %d epochs\n", loop);
        printf("- Final accuracy: %.1f%%\n", 100.0 * (pattern_count - error_count) / pattern_count);
        printf("- Average error per pattern: %.6f\n", error_total / pattern_count);

        return 1;  // Terminate training
    }

    // CONTINUE TRAINING - Reset counters for next epoch
    error_total = 0;  // Reset error accumulator
    error_count = 0;  // Reset error pattern counter

    return 0;  // Continue training
}