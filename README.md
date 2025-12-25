## Evaluation Framework

The model was evaluated using a balanced evaluation framework focusing on both predictive performance and real-time usability.
### Setup
Install dependencies using:
```bash
pip install -r requirements.txt


### Metrics Used

| Metric | Weight | Description |
|------|--------|-------------|
| Accuracy | 40% | Measures overall correctness of ticket classification across all categories. |
| Precision & Recall | 30% | Evaluates category-wise prediction quality by minimizing false positives and false negatives. |
| F1-Score | 20% | Harmonic mean of precision and recall, ensuring balanced performance on imbalanced data. |
| Latency | 10% | Measures response time of the API to ensure real-time applicability. |

### Evaluation Results

- **Accuracy:** ~80%  
- **Macro F1-Score:** ~0.82  
- **Precision & Recall:** Strong across most ticket categories, particularly for Billing Inquiry and Feature Request  
- **Latency:** < 100 ms per prediction via FastAPI endpoint  

### Observations

- The model demonstrates reliable classification performance despite a small, synthetic dataset.
- High precision ensures minimal misclassification, while strong recall captures most relevant tickets.
- Low latency makes the solution suitable for real-time customer support workflows.

Overall, the evaluation confirms that the system meets both performance and operational requirements for automated customer support ticket triaging.
