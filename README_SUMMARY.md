**Deep Learning Tuning Playbook Summary**

**Who is this document for?**

This document targets engineers and researchers aiming to maximize the performance of deep learning models, particularly through hyperparameter tuning. It assumes a foundational understanding of machine learning and focuses on supervised learning problems.

**Why a tuning playbook?**

The playbook addresses the significant guesswork and undocumented practices in tuning deep neural networks. It compiles the authors' extensive experience and aims to systematize experimental protocols in deep learning, encouraging the community to document and debate procedures to identify optimal practices.

**Guide for starting a new project**

**Choosing the model architecture:** Start with a proven model architecture and consider custom models later. Choose hyperparameters related to the model's size and structure, initially selecting a configuration that has demonstrated success in similar tasks.

**Choosing the optimizer:** Begin with popular optimizers for the problem type, such as SGD with momentum or Adam. Focus on simpler optimizers with fewer hyperparameters initially, and consider more complex optimizers later if needed.

**Choosing the batch size:** Select the largest batch size supported by hardware to enhance training speed. The batch size should not be used to tune validation performance directly but rather for computational efficiency.

**Choosing the initial configuration:** Set a simple, efficient configuration that achieves reasonable results. This initial setup should be easy to run and consume minimal resources, enabling effective hyperparameter tuning.

**A scientific approach to improving model performance**

**The incremental tuning strategy:** Begin with a basic configuration and incrementally add improvements based on strong evidence. The process involves identifying goals, designing experiments, learning from results, and potentially launching new configurations.

**Exploration vs exploitation:** Prioritize gaining insight into the problem through exploration before focusing on maximizing validation performance. This approach helps identify critical hyperparameters and potential new features while avoiding unnecessary complexity.

**Choosing the goal for the next round of experiments:** Each experiment should have a clear, narrow goal, such as testing a new regularizer or understanding a model hyperparameter's impact.

**Designing the next round of experiments:** Identify scientific, nuisance, and fixed hyperparameters for the experimental goal. Optimize nuisance hyperparameters while varying scientific ones to ensure fair comparisons.

**Extracting insight from experimental results:** Evaluate experiments to achieve their goals and answer additional questions to refine future experiments. Check for issues like search space boundaries, sampling density, and optimization failures.

**Determining whether to adopt a training pipeline change or hyperparameter configuration:** Consider variations in trial, study, and data collection when deciding to adopt changes. Run multiple trials to characterize trial variance and ensure improvements outweigh added complexity.

**After exploration concludes:** Use Bayesian optimization tools for final tuning once the search space is well-understood. Consider checking test set performance and potentially retraining with the validation set folded into the training set.

**Determining the number of steps for each training run**

**Deciding how long to train when training is not compute-bound:** Ensure training is long enough for the model to achieve the best possible result, adjusting the training step count based on the training and validation loss behavior.

**Deciding how long to train when training is compute-bound:** Balance the need for longer training with the benefits of running more experiments. Use multiple rounds of tuning with increasing training steps to gather insights and refine hyperparameters.

**Additional guidance for the training pipeline**

**Optimizing the input pipeline:** Use appropriate profiling tools to diagnose and address bottlenecks in the input pipeline, such as I/O latency or expensive preprocessing steps.

**Evaluating model performance:** Perform periodic evaluations during training using larger batch sizes and regular step intervals. Save sufficient information for offline analysis and ensure the evaluation set is representative.

**Saving checkpoints and retrospectively selecting the best checkpoint:** Periodically save model checkpoints and use retrospective optimal checkpoint selection to choose the best one based on validation performance.

**Setting up experiment tracking:** Maintain a tracking system for experiment results, including study details, trial counts, and validation performance, to facilitate reproducibility and analysis.

**Batch normalization implementation details:** Handle batch normalization carefully in multi-device settings, considering factors like ghost batch normalization and synchronization of exponential moving averages.

**Considerations for multi-host pipelines:** Ensure proper logging, checkpointing, and synchronization of batch norm statistics across hosts, and handle RNG seeds appropriately for initialization and data shuffling.

**FAQs**

**What is the best learning rate decay schedule family?** It's unclear, but having a schedule is crucial. Common choices include linear and cosine decay.

**Which learning rate decay should I use as a default?** Linear decay or cosine decay are preferred.

**Why do some papers have complicated learning rate schedules?** These are often tuned ad hoc based on validation performance. Replicating the schedule algorithm, rather than the schedule itself, is advisable.

**How should Adam’s hyperparameters be tuned?** Focus on tuning the learning rate first. For more trials, tune β1 and ε as well.

**Why use quasi-random search instead of more sophisticated black-box optimization algorithms during the exploration phase of tuning?** Quasi-random search offers consistent, reproducible results, allows post hoc analysis, and works well with high parallelism. It’s simpler and more robust for exploring hyperparameter spaces.

**Where can I find an implementation of quasi-random search?** Open-Source Vizier and MLCommons provide implementations based on Halton sequences.

**How many trials are needed to get good results with quasi-random search?** The number varies, but more trials generally improve results. Specific examples demonstrate the impact of trial count on performance.

**How can optimization failures be debugged and mitigated?** Identify instability through learning rate sweeps and gradient norm logging. Apply learning rate warmup, gradient clipping, or change optimizers as needed.

**Why do you call the learning rate and other optimization parameters hyperparameters?** Although "metaparameter" is more accurate, "hyperparameter" is widely used in the deep learning community.

**Why shouldn’t the batch size be tuned to directly improve validation set performance?** Once the training pipeline is optimized for each batch size, the batch size does not significantly impact maximum validation performance.

**What are the update rules for all the popular optimization algorithms?**

- **Stochastic gradient descent (SGD):** \(\theta*{t+1} = \theta*{t} - \eta_t \nabla \mathcal{l}(\theta_t)\)
- **Momentum:** \(v*0 = 0\), \(v*{t+1} = \gamma v*{t} + \nabla \mathcal{l}(\theta_t)\), \(\theta*{t+1} = \theta*{t} - \eta_t v*{t+1}\)
- **Nesterov:** \(v*0 = 0\), \(v*{t+1} = \gamma v*{t} + \nabla \mathcal{l}(\theta_t)\), \(\theta*{t+1} = \theta*{t} - \eta_t( \gamma v*{t+1} + \nabla \mathcal{l}(\theta\_{t}))\)
- **RMSProp:** \(v*0 = 1, m_0 = 0\), \(v*{t+1} = \rho v*{t} + (1 - \rho) \nabla \mathcal{l}(\theta_t)^2\), \(m*{t+1} = \gamma m*{t} + \frac{\eta_t}{\sqrt{v*{t+1} + \epsilon}}\nabla \mathcal{l}(\theta*t)\), \(\theta*{t+1} = \theta*{t} - m*{t+1}\)
- **ADAM:** \(m*0 = 0, v_0 = 0\), \(m*{t+1} = \beta*1 m*{t} + (1 - \beta*1) \nabla \mathcal{l} (\theta_t)\), \(v*{t+1} = \beta*2 v*{t} + (1 - \beta*2) \nabla \mathcal{l}(\theta_t)^2\), \(b*{t+1} = \frac{\sqrt{1 - \beta*2^{t+1}}}{1 - \beta_1^{t+1}}\), \(\theta*{t+1} = \theta*{t} - \alpha_t \frac{m*{t+1}}{\sqrt{v*{t+1}} + \epsilon} b*{t+1}\)
- **NADAM:** \(m*0 = 0, v_0 = 0\), \(m*{t+1} = \beta*1 m*{t} + (1 - \beta*1) \nabla \mathcal{l} (\theta_t)\), \(v*{t+1} = \beta*2 v*{t} + (1 - \beta*2) \nabla \mathcal{l} (\theta_t)^2\), \(b*{t+1} = \frac{\sqrt{1 - \beta*2^{t+1}}}{1 - \beta_1^{t+1}}\), \(\theta*{t+1} = \theta*{t} - \alpha_t \frac{\beta_1 m*{t+1} + (1 - \beta*1) \nabla \mathcal{l} (\theta_t)}{\sqrt{v*{t+1}} + \epsilon} b\_{t+1}\)
