Reviewer GPuR

We appreciate the feedback from the reviewer. We would like to add additional clarification.

## Weakness

> There are some errors in the mathematical formulations presented, particularly in equations (3) and (4), which are confusing.

Equation (3) is based on the original POMDP formulation by [Kaelbling et al., 1998](https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf) (page 107). In our variation, we introduce a multiplication term to account for a sequence of **physical exploration steps**. The timestep \( t \) in our model represents an **exploration sequence**. This modification allows us to encapsulate iterative exploration within a single belief update. If there are specific aspects of our formulation that appear incorrect, we would appreciate further clarification to address them appropriately.

Equation (4) is derived by replacing the traditional physical exploration components of the POMDP belief update with an imaginative exploration mechanism driven by a diffusion-based generative model parameterized by \( \theta \). In the standard belief update (Equation 3), the agent transitions between states \( s^t \) using the transition model \( T(s^{t+1} | s^t, a^t) \) and incorporates actual observations \( O(o^{t+1} | s^{t+1}, a^t) \), which requires summing over all possible prior states to account for uncertainty. By contrast, in imagination-driven belief revision, the agent remains in the current state \( s^t \) and employs the diffusion model \( p_{\hat{\theta}}(\hat{o}^{i+1} | \hat{o}^i, \hat{a}^i) \) to generate a sequence of hypothetical observations based on imagined actions \( \hat{a}^i \). This substitution eliminates the need for state transitions and the associated summations because the physical state does not change; instead, the belief is updated multiplicatively by the probabilities of the imagined observations across the imaginative steps \( I \). As a result, the belief \( \hat{b}^t(s^t) \) is directly refined by the product of these generated observation probabilities applied to the initial belief \( b^t(s^t) \), leading to Equation (4). This approach leverages the generative capabilities of the diffusion model \( \theta \) to simulate potential observations, enabling the agent to perform instantaneous and iterative belief revisions without altering the underlying state, thereby enhancing the efficiency and flexibility of the belief update process.

> Although SCL is highlighted as a contribution, its effectiveness is not demonstrated in the experimental results.

We compare the performance of SVD trained on Genex-DB against Genex trained using SCL on the same dataset, on common evaluation metrics:

| Model          | Input     | FVD ↓  | MSE ↓  | LPIPS ↓ | PSNR ↑  | SSIM ↑  |
|----------------|-----------|--------|--------|---------|---------|---------|
| SVD           | panorama  | 81.9   | 0.05   | 0.05   | 29.4    | 0.91  |
| **Genex**     | panorama  | **69.5** | **0.04** | **0.03** | **30.2** | **0.94** |

The difference in metrics is clear but not significant: Although edge inconsistencies are highly visible to humans, the metrics fail to fully capture such artifacts, as their impact is primarily localized to the edges of the image.

However, for exploration cycle consistency, the performance degradation is significant when the number of rotations increases.


> The use of latent diffusion with temporal attention is not a novel architecture.

We would like to clarify that our work does not emphasize the use of temporal attention as a core contribution. Our model is grounded in SVD, which we found sufficient for our task. The discussion of latent diffusion with temporal attention is included solely to explain the referenced work, which we have appropriately cited. This aspect was not intended to highlight our own contributions. We recognize the potential for misleading descriptions and will revise the text to ensure clarity and accuracy in this regard.



> The real-world dynamics of vehicles do not allow for pure rotation, which the paper seems to overlook. 5, Table 3 presents an unfair comparison.

We would like to clarify the context of Table 3 and address the concern about pure rotation.

Table 3 is intended to evaluate the generation quality of Genex by comparing it to other novel view synthesis methods. In this experiment, we place an object in the scene, use Genex to simulate forward movement, and evaluate the generated observation of this object from a new perspective (e.g., generating a high-quality back view given a front view). This process is an essential aspect of creating a coherent and realistic generated world. All the compared models are specifically designed and trained for cyclic or rotational novel view generation. This experiment does not involve views captured from a vehicle’s perspective. Could you elaborate on what makes this comparison "unfair"? We would be glad to provide additional clarification or further details regarding any concerns about this comparison.

In addition, one of the main motivations for using a panorama-based representation is its capacity for pure rotation, which significantly facilitates world exploration. While real-world vehicle dynamics do not allow for pure rotation, Genex’s effectiveness is highlighted by its ability to overcome this limitation with unlimited rotation and navigation. This enables agents to fully observe their surroundings, supporting more robust decision-making.

Finally, our work is not limited to navigation from the perspective of a vehicle. Genex is capable for almost all embodied scenarios, enabling imaginative exploration from the observation of a person, a car, or any other agent.




## Questions

> Is the LLM policy model fine-tuned or used as is?

In our experiments, the policy model is used as is, without any fine-tuning.

> It is unclear whether the diffusion model has been overfitted to the dataset, potentially making it inadequate for handling complex real-world interactions.

We have conducted extensive experiments to evaluate the generalizability of Genex. The numerical results are presented in Section 5.2 (Table 2), and the visual demonstrations are included in Appendix A.7 (Figure 18). Our results indicate that Genex, trained on synthetic data, demonstrates robust zero-shot generalizability to real-world scenarios. Specifically, the model trained on synthetic data performs well on scenes such as indoor behavior vision suites, outdoor Google Maps Street View in real-world settings, and other synthetic scenes that all differ significantly from the training distribution, without requiring additional fine-tuning.


> The space of 'state' & 'belief' is not clearly defined.
The state is the environment the agent is currently situated in, and the space is the entire world. As described in Equation (4), we remove the transition of states, simplifying the definition. At the beginning, the state is represented as a 3D environment used to sample the initial observation.
The belief operates at a higher level and pertains to the LLM's internal reasoning. Through continuous prompts in multi-hop conversations, the language model continually revises its belief about the world. These beliefs are encoded within the model's internal parameters, evolving as new observations and prompts are processed. Additionally, if we ask the agent to explicitly state its belief, the space would be represented in natural language.


> The entire framework appears to have little connection with POMDP.

While our framework does not strictly adhere to the traditional POMDP formalism, it fundamentally builds upon its core principles. Specifically, the state in our framework corresponds to the agent's environment, while the belief represents the agent’s internal reasoning and its evolving understanding of the world based on observations. Unlike standard POMDPs, which require physical exploration of the environment to update beliefs and gather new information, we replace that component with Genex. By enabling mental simulation and navigation, Genex streamlines the belief-updating process, significantly reducing the time and resource demands of physical exploration. This abstraction integrates reasoning and decision-making within complex, unstructured environments, fully leveraging and extending the foundational ideas of POMDPs.




If this does not fully address your concerns, we would appreciate further elaborations.
