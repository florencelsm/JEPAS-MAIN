# JEPAs
Un-official PyTorch implementations of:
- [x] I-JEPA: [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243)
- [x] V-JEPA: [Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/abs/2404.08471)
- [x] T-JEPA: Self-Supervised Learning from Text with a Joint-Embedding Predictive Architecture (Original Works)
- [ ] (In progress) A-JEPA: Self-Supervised Learning from Audio with a Joint-Embedding Predictive Architecture (Original Works)
- [ ] MC-JEPA: [MC-JEPA: A Joint-Embedding Predictive Architecture for Self-Supervised Learning of Motion and Content Features](https://arxiv.org/abs/2307.12698)
- [ ] Graph-JEPA: [Graph-level Representation Learning with Joint-Embedding Predictive Architectures](https://arxiv.org/abs/2309.16014)

## JEPAs Explained
Joint-Embedding Predictive Architectures (JEPAs) utilise latent space representations to learn rich semantic understandings of inputs.
This is achieved via a method called **self-distillation** - in which a model's latent space representations form its own prediction targets, thus supervising its own learning.
Many self-supervised techniques learn from the *structure implicit in data signals,* rather than relying on labels for supervision.
The JEPA paradigm takes this one step further - it incentivises the learning of progressively more expressive models (interpretations and dynamics) of input data signals by tasking itself with *learning from the structure of its internal model of those input data signals:*
  After interpreting a signal, the model tests itself by predicting *its own interpretation of the input signal* - thus it learns to structure its percepts, and learns to the structure of its internal model of the structured external world - developing a progressively more sophisticated "world model".

