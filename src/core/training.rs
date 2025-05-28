use serde::{Deserialize, Serialize};

/// Training pattern for ED learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingPattern {
    /// Input values (will be doubled for excitatory/inhibitory pairs)
    pub inputs: Vec<f64>,
    /// Target output values
    pub targets: Vec<f64>,
    /// Pattern identifier
    pub id: usize,
}

impl TrainingPattern {
    /// Create new training pattern
    pub fn new(inputs: Vec<f64>, targets: Vec<f64>, id: usize) -> Self {
        Self {
            inputs,
            targets,
            id,
        }
    }

    /// Create XOR training dataset
    pub fn create_xor_dataset() -> Vec<Self> {
        vec![
            Self::new(vec![0.0, 0.0], vec![0.0], 0),
            Self::new(vec![1.0, 0.0], vec![1.0], 1),
            Self::new(vec![0.0, 1.0], vec![1.0], 2),
            Self::new(vec![1.0, 1.0], vec![0.0], 3),
        ]
    }

    /// Create N-bit parity dataset
    pub fn create_parity_dataset(n_bits: usize) -> Vec<Self> {
        let mut patterns = Vec::new();
        let n_patterns = 1 << n_bits; // 2^n_bits

        for i in 0..n_patterns {
            let mut inputs = Vec::with_capacity(n_bits);
            let mut parity = 0;

            // Generate binary pattern
            for bit in 0..n_bits {
                let value = if (i >> bit) & 1 == 1 {
                    1.0
                } else {
                    0.0
                };

                inputs.push(value);

                if value > 0.5 {
                    parity += 1;
                }
            }

            // Parity target: 1 if odd number of 1's, 0 if even
            let target = if parity % 2 == 1 {
                1.0
            } else {
                0.0
            };

            patterns.push(Self::new(inputs, vec![target], i));
        }

        patterns
    }
}
