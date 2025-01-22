# AskTheChef
A Deep Learning Based Chatbot

This project explores the potential of deep-learning techniques to enhance a recipe chatbot's ability to provide customized and user-centric recipe suggestions, addressing a critical gap in existing tools and methods. To answer the question: How can deep learning techniques enhance a recipe chatbot's ability to understand user queries and provide accurate, contextually relevant recipe recommendations?

Dataset: https://huggingface.co/datasets/m3hrdadfi/recipe_nlg_lite

Recipe Generator Model: https://huggingface.co/mbien/recipenlg

## Approach

To address the question, a novel recipe chatbot framework was developed by leveraging deep learning techniques for intent classification, recipe generation, and conversational interaction. This section describes the methodological framework, including model architecture, data preparation, training, system integration, and evaluation.

### Model Architecture

The chatbot was designed with three core components: an intent classifier, a recipe generator, and a conversational model. The intent classifier serves as the entry point, distinguishing between user queries usage for recipe generation and general conversational dialogue. By directing user inputs to the appropriate subsystem, the classifier ensures contextual relevance and precision in responses.

The recipe generator utilizes a pre-trained language model fine-tuned on a curated recipe dataset to craft detailed and accurate recipes. This subsystem is designed to understand ingredient-specific or cooking-method-related queries, translating them into coherent recipe outputs. On the other hand, the conversational model manages interactions unrelated to recipe generation, ensuring natural and engaging dialogue. Both the recipe generator and conversational model were built using transformer-based architectures such as DistilGPT-2, GPT-2, and Llama.

### Data Preprocessing

For the intent classifier, a pre-trained zero-shot classification model was employed to differentiate recipe-related queries (e.g., “How do I cook pasta?”) from general conversational ones (e.g., “How are you?”). Specifically, the facebook/bart-large-mnli model, accessed via the Transformers library’s pipeline, was used. This model enables classification without requiring labeled training data, leveraging its pre-trained capabilities to associate user inputs with predefined labels, such as recipe_request and general_conversation.

In the recipe generation subsystem, datasets were preprocessed to maintain uniformity in ingredient lists, preparation steps, and recipe formats. Tokenization was performed using pre-trained language model tokenizers, and recipes were segmented into context and output pairs, where the input was a user query and the output was a recipe.

Conversational data was similarly preprocessed to fine-tune the conversational model. This involved cleaning and standardizing text from open-domain dialogue datasets to improve the chatbot’s fluency and ability to handle diverse user queries. 

### Model Usage, System Integration and Implementation

The recipe chatbot leveraged pre-trained models for all its components, avoiding the need for additional training or fine-tuning. The intent classifier used the facebook/bart-large-mnli model, a zero-shot classification system accessible via the Transformers library. This pre-trained model was configured to classify user queries into predefined categories such as recipe_request and general_conversation, ensuring robust intent detection without requiring labeled datasets.

The recipe generator utilized the pre-trained mbien/recipenlg model from Hugging Face's Transformers library. This model was specifically designed for generating recipes and was applied as-is. Query inputs were tokenized and fed into the model to generate recipes tailored to the user’s request. The outputs were formatted to provide clear, step-by-step instructions and ingredient lists.

For the conversational model, three candidate pre-trained generative models were evaluated: Llama, DistilGPT-2, and GPT-2. These models were applied without fine-tuning to handle user queries unrelated to recipes. Configuration of generation parameters, such as maximum response length, temperature, repetition penalty, and sampling strategies (e.g., top-k and nucleus sampling), ensured the generated responses were both creative and contextually appropriate.

These pre-trained models formed the foundation of the chatbot system, eliminating the need for fine-tuning or additional training. Queries were routed through the intent classifier, which directed them to the appropriate subsystem: either the recipe generator for recipe-related queries or the conversational model for general dialogue. The system was modularly designed, allowing seamless integration and dynamic response rendering.

Implementation relied on the Hugging Face Transformers library for leveraging pre-trained models. Custom configurations were used to optimize outputs without requiring computationally expensive retraining. These optimizations included fine-tuning hyperparameters such as response length, nucleus sampling, and temperature to balance creativity and relevance in generated responses. The modularity also allowed experimentation with different chatbot models to determine the best-performing one for conversational tasks.

### Performance Metrics

The chatbot’s performance was evaluated across three dimensions: accuracy, relevance, and user experience. Accuracy was measured using BLEU and ROUGE scores, comparing generated recipes against reference outputs. For relevance, semantic similarity scores were computed to gauge how closely the chatbot’s responses aligned with user expectations. 

User experience was assessed through simulated interactions, where coherence, fluency, and engagement of dialogue responses were qualitatively analyzed. These evaluations provided insights into the strengths and limitations of each subsystem, guiding further refinements.
