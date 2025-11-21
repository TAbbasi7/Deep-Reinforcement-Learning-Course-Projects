# One-Shot Learning with Siamese Networks 👯

Implementation of a Siamese Network using Contrastive Loss to distinguish between digit pairs (similar vs. dissimilar). This approach is crucial for One-Shot Learning tasks.

## 🔬 Critical Findings & Hyperparameter Tuning

We conducted extensive experiments to find the stable configuration.

| Config | LR | Dropout | Result | Analysis |
| :--- | :--- | :--- | :--- | :--- |
| **Winner** 🏆 | `0.001` | **0.5** | **96.23%** | High dropout stabilized the aggressive learning rate. |
| Failure | `0.001` | 0.3 | NaN | Suffered from **Exploding Gradient** due to insufficient regularization. |

## Conclusion
The experiment proves that for this specific architecture, a higher Dropout rate (0.5) was strictly necessary to prevent numerical instability.
## 📊 Results
![Siamese Network Training](./Siamese_Network_Results.png)
