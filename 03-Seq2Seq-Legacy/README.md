# Legacy Seq2Seq Chatbot Analysis 🤖

An experimental implementation of a basic LSTM-based Encoder-Decoder chatbot.

## ⚠️ Project Status: Educational Analysis
**Outcome:** The model failed to generate meaningful responses (e.g., input "how are you" -> output "i am n").

## Why did it fail?
This project serves as a demonstration of the limitations of vanilla Seq2Seq models trained on tiny datasets:
1.  **Data Scarcity:** Trained on only 5 sample pairs (insufficient for generalization).
2.  **Architecture Limits:** Basic LSTM without Attention mechanisms struggles with context retention.
3.  **Conclusion:** For functional conversational AI, modern architectures like **Transformers** and large-scale pre-training (LLMs) are required.