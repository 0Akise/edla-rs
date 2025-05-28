use edla_rs::core::{
    neuron::{ErrorChannels, NeuronType},
    training::TrainingPattern,
    utils::sigmoid,
};

#[test]
fn test_neuron_type_alternation() {
    assert_eq!(NeuronType::from_index(0), NeuronType::Inhibitory);
    assert_eq!(NeuronType::from_index(1), NeuronType::Excitatory);
    assert_eq!(NeuronType::from_index(2), NeuronType::Inhibitory);
    assert_eq!(NeuronType::from_index(3), NeuronType::Excitatory);
}

#[test]
fn test_error_channel_creation() {
    let positive_error = ErrorChannels::from_prediction_error(0.5);
    assert_eq!(positive_error.excitatory, 0.5);
    assert_eq!(positive_error.inhibitory, 0.0);

    let negative_error = ErrorChannels::from_prediction_error(-0.3);
    assert_eq!(negative_error.excitatory, 0.0);
    assert_eq!(negative_error.inhibitory, 0.3);
}

#[test]
fn test_xor_dataset_creation() {
    let xor_data = TrainingPattern::create_xor_dataset();
    assert_eq!(xor_data.len(), 4);

    assert_eq!(xor_data[0].targets[0], 0.0); // [0,0] -> 0
    assert_eq!(xor_data[1].targets[0], 1.0); // [1,0] -> 1  
    assert_eq!(xor_data[2].targets[0], 1.0); // [0,1] -> 1
    assert_eq!(xor_data[3].targets[0], 0.0); // [1,1] -> 0
}

#[test]
fn test_sigmoid_function() {
    let result = sigmoid(0.0, 0.4);
    assert!((result - 0.5).abs() < 1e-10); // sigmoid(0) should be 0.5

    let positive = sigmoid(1.0, 0.4);
    assert!(positive > 0.5);

    let negative = sigmoid(-1.0, 0.4);
    assert!(negative < 0.5);
}
