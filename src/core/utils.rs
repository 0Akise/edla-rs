use rand::Rng;

/// Sigmoid activation function with configurable steepness
/// Formula: 1 / (1 + exp(-2*x/steepness))
pub fn sigmoid(x: f64, steepness: f64) -> f64 {
    1.0 / (1.0 + (-2.0 * x / steepness).exp())
}

/// Sigmoid derivative for weight updates
/// Formula: sigmoid(x) * (1 - sigmoid(x))
pub fn sigmoid_derivative(output: f64) -> f64 {
    output * (1.0 - output)
}

/// Generate random weight within specified range
pub fn random_weight<R: Rng>(rng: &mut R, range: f64) -> f64 {
    rng.random::<f64>() * range
}
