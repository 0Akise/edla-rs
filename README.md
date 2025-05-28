# edla-rs: Error Diffusion Learning Algorithm for Neural Networks
### *A Forgotten Method, Alternative to Backpropagation - Resurrected from 1999*

> **"…十分な中間ユニットがあればED法ではXORを5ステップで学習できることが分かります。"**
>
> **"...Here, we can see that ED can solve XOR in just 5 steps, if there are enough hidden layers"**
>
> *— Isamu Kaneko, 1999 in his website after experiments*


## ⏰ Current Status of repo
- [x] Refactor Kaneko's cryptic original codebase (see `ed_original` and `ed_reworked` folder)
- [ ] *Migrate refactored version into Rust*
- [ ] Implement MNIST Test
- [ ] Write `CONTRIBUTING.md`
- [ ] Investigate integrations with other network architectures(CNN, Transformer etc.)


## 🧠 What is Error Diffusion Learning?
**Error Diffusion (ED)** is a revolutionary neural network learning algorithm, developed by **Isamu Kaneko** in 1999-2000 that replaces backpropagation's complex gradient calculations with a simple, biologically-plausible mechanism inspired by neurotransmitter diffusion in the brain.


### 💡 The Key Insight 
Instead of computing different gradients for each layer (like backpropagation), ED broadcasts the **same error signal** to ALL layers simultaneously - like how dopamine affects multiple brain regions at once.

```rust
// Backpropagation: Complex layer-specific gradients
layer3_gradient = error × chain_rule_layer3
layer2_gradient = error × chain_rule_layer3 × chain_rule_layer2  
layer1_gradient = error × chain_rule_layer3 × chain_rule_layer2 × chain_rule_layer1

// Error Diffusion: Same signal to all layers
all_layers_signal = error
```


## 🔬 How It Works
### 1. **Neuron Types**
Each neuron is either **Excitatory (+)** or **Inhibitory (-)** in alternating pattern:
```
Input Layer:  [+] [-] [+] [-] ...
Hidden Layer: [-] [+] [-] [+] ...
Output Layer: [+] (always excitatory)
```

### 2. **Error Channel Splitting**
```rust
if prediction_error > 0 {
    excitatory_channel = error;  // Need to increase output
    inhibitory_channel = 0;
}
else {
    excitatory_channel = 0;
    inhibitory_channel = -error;  // Need to decrease output
}
```

### 3. **Uniform Broadcasting**
The **same** error signal diffuses to ALL hidden neurons simultaneously - no layer-by-layer propagation needed!

### 4. **Type-Constrained Learning**
Weight updates depend on neuron type combinations:
- **Excitatory → Excitatory**: Strengthen connection
- **Inhibitory → Inhibitory**: Strengthen connection  
- **Excitatory → Inhibitory**: Weaken connection
- **Inhibitory → Excitatory**: Weaken connection


## 📊 Benchmarks (Kaneko's Original Results, 1999)
### XOR Problem - Convergence Steps
| Hidden Units | Error Diffusion | Backpropagation |
|--------------|----------------|-----------------|
| 8 | **75 steps** | 137 steps |
| 16 | **28 steps** | 127 steps |
| 32 | **8 steps** | 117 steps |
| 64 | **6 steps** | 94 steps |
| 128+ | **5 steps** | **Failed** |

*ED achieves ~5 step convergence with sufficient hidden units, while BP fails on very deep networks*

### N-bit Parity Problems
| Input Bits | ED (64 hidden) | BP (64 hidden) |
|------------|----------------|----------------|
| 2-bit (XOR) | **6 steps** | 190 steps |
| 3-bit | **11 steps** | **Failed** |
| 4-bit | **56 steps** | **Failed** |
| 5-bit | **331 steps** | **Failed** |

*BP often cannot solve parity problems at all, while ED scales reliably*

### Hand-written Digit Recognition (16×16 images, 1000 training samples)
| Method | Accuracy | Training Speed |
|--------|----------|----------------|
| **Error Diffusion** | 89-93% | **9-21 steps** |
| **Backpropagation** | 94-95% | 74-225 steps |

*Trade-off: BP has slightly better accuracy, but ED trains 10× faster*


## 🚀 Why This Matters
### **The Trade-offs**
| Advantage | Error Diffusion | Backpropagation |
|-----------|----------------|-----------------|
| **Logic Problems** | **Dominant** (5 steps vs 100+) | Often fails |
| **Deep Networks** | **No vanishing gradients** | Gradient problems |
| **Training Speed** | **~10× faster** | Slower but precise |
| **Image Recognition** | Good (89-93%) | **Better** (94-95%) |
| **Parameter Sensitivity** | **Robust** | Requires tuning |

### **ED's True Strength**
ED isn't universally better - it's **fundamentally different**. While BP optimizes through precise gradient calculations, ED uses biologically-plausible error diffusion. This makes ED:
- **Exceptionally fast** at logic problems  
- **Robust** to network depth and parameters
- **Biologically realistic** for understanding brain learning
- **Complementary** to existing methods, not necessarily replacing them

### **The Lost Algorithm**
Kaneko's work was **25 years ahead of its time**, proposing solutions to problems that only became widely recognized much later:
- **1999**: ED solves vanishing gradients, excels at logic problems
- **2010s**: Deep learning community "discovers" vanishing gradient problem
- **2024**: Researchers on Qiita community rediscover ED, one reporting 98% accuracy achievement on MNIST binary classification


## 🧬 Biological Inspiration
ED mimics how real brains might learn:

| Biological Reality | ED Implementation |
|-------------------|-------------------|
| Neurotransmitter diffusion | Error signal broadcasting |
| Excitatory/inhibitory neurons | `NeuronType::Excitatory/Inhibitory` |
| No "backwards" signals | No gradient backpropagation |
| Parallel synaptic updates | Simultaneous weight updates |

## 🏛️ Historical Context
### The Timeline
- **1999-**: Kaneko develops ED at height of "AI Winter" and publishes algorithm with C implementation
- **2006-**: Deep learning revolution begins (Hinton et al.)
- **2010-**: Vanishing gradient problem widely recognized
- **2024-**: Japanese researchers rediscover ED
- **2025-**: This repository resurrects and aims to modernize the algorithm

### The Tragedy
Kaneko passed away in 2014 before completing this research, and his revolutionary algorithm was lost to time. This implementation honors his memory and brings his insights to modern AI research.

## 🔬 Research Applications
### Where ED Excels
- **Logic Problems**: XOR, parity, boolean functions
- **Deep Networks**: No vanishing gradients
- **Rapid Prototyping**: Robust default parameters
- **Neuromorphic Computing**: Biologically-inspired hardware
- **Educational**: Understanding learning without calculus

### Current Research Directions
- [ ] Convolutional ED networks
- [ ] Transformer architecture with ED
- [ ] Comparison with modern methods
- [ ] Theoretical analysis of convergence properties

## 🤝 Contributing
We welcome contributions to honor Kaneko's legacy! Areas of interest:
- **Algorithm Improvements**: Modernizing ED for current problems
- **Theoretical Analysis**: Mathematical foundations and proofs
- **Benchmarking**: Comprehensive performance studies
- **Documentation**: Tutorials and educational materials

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.


## 📖 Citation
If you use this implementation in your research, please cite:
```bibtex
@misc{kaneko1999ed,
  title={Error Diffusion Learning Algorithm for Neural Networks},
  author={Kaneko, Isamu},
  year={1999},
  url={https://web.archive.org/web/19991124023203/http://village.infoweb.ne.jp/~fwhz9346/ed.htm},
  note={archived version of original implementation.}
}
```


## 🙏 Acknowledgments
- **Isamu Kaneko (金子 勇)** (1970-2014) - Original algorithm creator and author of original code
- **Qiita community** - 2024 rediscovery
- **Internet Archive** - Preserving Kaneko's original work


## ⚖️ License
Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
This permissive license ensures Kaneko's revolutionary algorithm can benefit the widest possible research community.


## 🌟 Help Spread His Legacy!
If this project helped you discover the power of biologically-inspired learning, please consider starring it to help others find this hidden gem of neural network history!


## 🔗 Reference Links
- [Isamu Kaneko's Original Work](https://web.archive.org/web/19991124023203/http://village.infoweb.ne.jp/~fwhz9346/ed.htm)
- [Qiita | 『Winny』の金子勇さんの失われたED法を求めて...いたら見つかりました](https://qiita.com/kanekanekaneko/items/901ee2837401750dfdad)
- [Qiita | 金子勇さんのED法を実装してMNISTを学習させてみた](https://qiita.com/pocokhc/items/f7ab56051bb936740b8f)
- [Qiita | 金子勇さんのED法の解説と弱点、行列積を使用した効率的な実装](https://qiita.com/Cartelet/items/a18e32348adc0c689db4)
- ... and many others.