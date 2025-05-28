use std::fmt;

use serde::{Deserialize, Serialize};

use super::utils::sigmoid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeuronType {
    /// Excitatory neurons strengthen connections when activated
    Excitatory,
    /// Inhibitory neurons weaken connections when activated
    Inhibitory,
}

impl NeuronType {
    /// Convert neuron type to mathematical weight multiplier (Excitatory = +1.0, Inhibitory = -1.0)
    pub fn as_weight_factor(&self) -> f64 {
        match self {
            NeuronType::Excitatory => 1.0,
            NeuronType::Inhibitory => -1.0,
        }
    }

    /// Create alternating excitatory/inhibitory pattern
    pub fn from_index(index: usize) -> Self {
        if (index + 1) % 2 == 0 {
            NeuronType::Excitatory
        } else {
            NeuronType::Inhibitory
        }
    }
}

impl fmt::Display for NeuronType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NeuronType::Excitatory => write!(f, "+"),
            NeuronType::Inhibitory => write!(f, "-"),
        }
    }
}

/// Error signal channels for Error Diffusion learning
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ErrorChannels {
    pub excitatory: f64,
    pub inhibitory: f64,
}

impl ErrorChannels {
    /// Create new error channels from prediction error (Positive → excitatory, negative → inhibitory)
    pub fn from_prediction_error(error: f64) -> Self {
        if error > 0.0 {
            Self {
                excitatory: error,
                inhibitory: 0.0,
            }
        } else {
            Self {
                excitatory: 0.0,
                inhibitory: -error,
            }
        }
    }

    /// Check if any error signal is present
    pub fn has_error_signal(&self) -> bool {
        self.excitatory > 0.0 || self.inhibitory > 0.0
    }

    /// Get the dominant error magnitude
    pub fn error_magnitude(&self) -> f64 {
        self.excitatory.max(self.inhibitory)
    }
}

impl fmt::Display for ErrorChannels {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "E:{:.4} I:{:.4}", self.excitatory, self.inhibitory)
    }
}

/// Individual neuron state within the ED network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neuron {
    /// Neuron type (excitatory or inhibitory)
    pub neuron_type: NeuronType,
    /// Current input activation level
    pub input: f64,
    /// Current output activation level  
    pub output: f64,
    /// Error signals for this neuron (excitatory/inhibitory channels)
    pub error_channels: ErrorChannels,
    /// Neuron index within the network
    pub index: usize,
}

impl Neuron {
    /// Create new neuron with specified type and index
    pub fn new(neuron_type: NeuronType, index: usize) -> Self {
        Self {
            neuron_type,
            input: 0.0,
            output: 0.0,
            error_channels: ErrorChannels::default(),
            index,
        }
    }

    /// Apply sigmoid activation function
    pub fn activate(&mut self, steepness: f64) {
        self.output = sigmoid(self.input, steepness);
    }

    /// Reset neuron state for new pattern
    pub fn reset(&mut self) {
        self.input = 0.0;
        self.output = 0.0;
        self.error_channels = ErrorChannels::default();
    }

    /// Check if neuron is excitatory
    pub fn is_excitatory(&self) -> bool {
        matches!(self.neuron_type, NeuronType::Excitatory)
    }

    /// Check if neuron is inhibitory
    pub fn is_inhibitory(&self) -> bool {
        matches!(self.neuron_type, NeuronType::Inhibitory)
    }
}

/// Connection weight between two neurons in the ED network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    /// Source neuron index
    pub from: usize,
    /// Target neuron index
    pub to: usize,
    /// Connection weight value (constrained by neuron types)
    pub weight: f64,
    /// Whether this connection is enabled
    pub connection_enabled: bool,
}

impl Connection {
    /// Create new connection with ED neuron type constraints applied
    pub fn new(from: usize, to: usize, base_weight: f64, from_type: NeuronType, to_type: NeuronType) -> Self {
        let constrained_weight = base_weight * from_type.as_weight_factor() * to_type.as_weight_factor();

        Self {
            from,
            to,
            weight: constrained_weight,
            connection_enabled: true,
        }
    }

    /// Update weight using ED learning rule
    pub fn update_ed_weight(&mut self, delta_base: f64, error_signal: f64, from_type: NeuronType, to_type: NeuronType) {
        if self.connection_enabled {
            let weight_delta = delta_base * error_signal * from_type.as_weight_factor() * to_type.as_weight_factor();

            self.weight += weight_delta;
        }
    }
}
