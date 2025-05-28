use std::fmt;

use serde::{Deserialize, Serialize};

use super::neuron::{Connection, Neuron, NeuronType};
use super::training::TrainingPattern;

/// Type of network layer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerType {
    /// Input layer (doubled for excitatory/inhibitory pairs)
    Input,
    /// Hidden processing layer
    Hidden,
    /// Output layer
    Output,
    /// Bias/threshold layer
    Bias,
}

/// Network configuration flags controlling ED learning behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Number of recurrent processing timesteps per forward pass
    pub timesteps: usize,
    /// Learning rate controlling weight update magnitude
    pub learning_rate: f64,
    /// Global bias input applied to all neurons
    pub bias: f64,
    /// Sigmoid activation function steepness parameter
    pub sigmoid_steepness: f64,
    /// Error amplification factor for hidden layers
    pub error_amplification: f64,
    /// Weight initialization range
    pub weight_init_range: f64,
    /// Threshold/bias initialization range  
    pub threshold_init_range: f64,
    /// Residual error threshold for convergence detection
    pub convergence_threshold: f64,
    /// Enable multi-layer processing (prevents input shortcuts)
    pub flag_multilayer: bool,
    /// Cut recurrent loops for more feedforward structure
    pub flag_loop_cutting: bool,
    /// Prevent neuron self-connections
    pub flag_self_loop_cutting: bool,
    /// Enable inhibitory input connections
    pub flag_inhibitory_inputs: bool,
    /// Enable bidirectional error application mode
    pub mode_weight_decrement: bool,
}

impl Default for NetworkConfig {
    /// Default ED parameters based on Kaneko's research
    fn default() -> Self {
        Self {
            timesteps: 2,
            learning_rate: 0.8,
            bias: 0.8,
            sigmoid_steepness: 0.4,
            error_amplification: 1.0,
            weight_init_range: 1.0,
            threshold_init_range: 1.0,
            convergence_threshold: 0.1,
            flag_multilayer: true,
            mode_weight_decrement: false,
            flag_loop_cutting: true,
            flag_self_loop_cutting: true,
            flag_inhibitory_inputs: true,
        }
    }
}

/// Network layer containing neurons and their properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkLayer {
    /// Neurons in this layer
    pub neurons: Vec<Neuron>,
    /// Layer type identifier
    pub layer_type: LayerType,
    /// Layer index in network
    pub layer_index: usize,
}

impl NetworkLayer {
    /// Create new layer with specified type and size
    pub fn new(layer_type: LayerType, size: usize, layer_index: usize) -> Self {
        let mut neurons = Vec::with_capacity(size);

        for i in 0..size {
            let neuron_type = match layer_type {
                LayerType::Output => NeuronType::Excitatory, // Output always excitatory
                _ => NeuronType::from_index(i),              // Alternating pattern for others
            };
            neurons.push(Neuron::new(neuron_type, i));
        }

        Self {
            neurons,
            layer_type,
            layer_index,
        }
    }

    /// Reset all neurons in this layer
    pub fn reset(&mut self) {
        for neuron in &mut self.neurons {
            neuron.reset();
        }
    }

    /// Get neuron by index safely
    pub fn get_neuron(&self, index: usize) -> Option<&Neuron> {
        self.neurons.get(index)
    }

    /// Get mutable neuron by index safely
    pub fn get_neuron_mut(&mut self, index: usize) -> Option<&mut Neuron> {
        self.neurons.get_mut(index)
    }
}

/// Learning statistics for monitoring ED training progress
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LearningStats {
    /// Current epoch number
    pub epoch: usize,
    /// Total accumulated error
    pub total_error: f64,
    /// Number of patterns with significant error
    pub error_count: usize,
    /// Total number of patterns
    pub pattern_count: usize,
    /// Learning curve history
    pub error_history: Vec<f64>,
    /// Convergence achieved flag
    pub converged: bool,
    /// Final accuracy percentage
    pub accuracy: f64,
}

impl LearningStats {
    /// Create new learning statistics
    pub fn new(pattern_count: usize) -> Self {
        Self {
            pattern_count,
            ..Default::default()
        }
    }

    /// Update statistics for current epoch
    pub fn update_epoch(&mut self, epoch: usize, total_error: f64, error_count: usize) {
        self.epoch = epoch;
        self.total_error = total_error;
        self.error_count = error_count;
        self.error_history.push(total_error);
        self.accuracy = 100.0 * (self.pattern_count - error_count) as f64 / self.pattern_count as f64;
    }

    /// Check if learning has converged
    pub fn check_convergence(&mut self, threshold: f64) -> bool {
        self.converged = self.total_error < threshold;
        self.converged
    }

    /// Get current error rate
    pub fn error_rate(&self) -> f64 {
        self.error_count as f64 / self.pattern_count as f64
    }
}

impl fmt::Display for LearningStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Epoch:{} Error:{:.6} Accuracy:{:.1}% Patterns:{}/{}",
            self.epoch,
            self.total_error,
            self.accuracy,
            self.pattern_count - self.error_count,
            self.pattern_count
        )
    }
}

/// Network dimensional parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkDimensions {
    /// Number of input neurons (logical inputs, will be doubled)
    pub input_size: usize,
    /// Number of hidden neurons
    pub hidden_size: usize,
    /// Number of output neurons
    pub output_size: usize,
    /// Total number of neurons
    pub total_neurons: usize,
}

impl NetworkDimensions {
    /// Create new network dimensions
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let total_neurons = 2 + (input_size * 2) + hidden_size + output_size; // +2 for bias

        Self {
            input_size,
            hidden_size,
            output_size,
            total_neurons,
        }
    }
}

/// Main Error Diffusion Neural Network structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EDNetwork {
    /// Network layers (bias, input, hidden, output)
    pub layers: Vec<NetworkLayer>,
    /// Connection matrix between all neurons
    pub connections: Vec<Vec<Connection>>,
    /// Network configuration parameters
    pub config: NetworkConfig,
    /// Network dimensions
    pub dimensions: NetworkDimensions,
    /// Current learning statistics
    pub stats: LearningStats,
    /// Training patterns
    pub training_data: Vec<TrainingPattern>,
}
