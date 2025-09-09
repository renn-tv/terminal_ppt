# LLM Intro and HowTo
* Goal: 
* get a basic understanding of the LLM blackbox and use that knowledge for better interaction with AI tools.
## How LLM is created
### Base Model
1) Gather and prepare data  <!-- curated internet, de-duplication, Wikipedia, books, code, licenced data, domain coverage, .. Dataset quality is more important than quantity. Watch for safety risks and biases. -->
2) Tokenization based on Vocabulary <!-- 50.000 - 150.000 entries -->  
3) Large Scale Training, Neural Network <!-- Parallelization, distributed training, based on architecture. -->
4) Inference and next word prediction <!-- conditional probability -->
### SFT Model
* Post-Training as Supervised Finetuning 
* Examples drafted by human experts   
* prepared datasets from human input    
* Helpful Assistant, answers questions    
<!-- SFT model aligns base models with human intent (Q&A, Assistant behaviour) -->
### RL Model
* Post-Training: Reinforcement Learning
   1) Learns "solving" tasks, from problem to answer
   2) discovers "thinking" and "strategies" <!-- meaning they optimize toward human-preferred outcomes -->
   3) difficult in un-verifiable domains like humor   
   * RLHF = RL from Human Feedback, ranking by human   
   * but: RL discovers ways to "game" the model  
   * Assistant can "reason", pattern emulated <!-- not conscious logic. Trade off: alignment vs loss of diversity & creativity -->
### Released Model
* released Model is a Fixed Mathematical Function <!-- parameters don't change, unless retrained or updated -->
## Technical Details
### Tokenization
  * To save compute and gain efficiency
  * [Tiktokenizer](https://tiktokenizer.vercel.app/?model=cl100k_base) and [Vocabulary](https://huggingface.co/BEE-spoke-data/cl100k_base/raw/main/tokenizer.json)
### Architecture
  #### Transformer
  * [Neural Network](https://playground.tensorflow.org/) (Playground)
  * [Brendon Bycroft](https://bbycroft.net/llm): inside LLM
  * [Difficult](https://pytorch.org/blog/inside-the-matrix/) to visualise full model
  * [Next Word Prediction](https://moebio.com/mind/)
  * [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) <!-- Attention = finding words that matter in context -->
  * [Parameter Count](https://huggingface.co/xai-org/grok-2/discussions/24)
  #### Hosting
  * Proprietary Models like ChatGPT 5 (OpenAI), Gemini 2.5 Pro (Google), Claude Opus 4.1 (Anthropic)
  * Open Weights Models like Gemma 3 (Google), gpt-oss (OpenAI)
  * Open Source Models (also contain training data and training process)
## How to use
### Where LLM is bad and why
* Counting 
* Spelling
* Detailed Knowledge
### Where LLM is good and why
* Writing: styles, grammar / translation / summaries / re-phrasing / explaining
* Coding, Programming
* General knowledge in basically all domains
* Brainstorming
* Learning
* Role Playing like expert or beginner
### Best Practice for LLM Interaction
* Spend time to develop a good prompt (aka Prompt Engineering) = context <!-- Prompt Engineering -->
* Use English - this is the language models see most during training
* Keep log for important prompts <!-- Our Ford LLM saves them too -->
* Create Assistant for repetitive prompts/work <!-- only a little bit more effort than a prompt -->
* Do not ask for analysis result, ask for code: transparent, re-use, company asset <!-- develop code to calculate the result. more transparent / allows re-use / can be documented as company asset --> 
* Don't argue with LLM, but iterate <!-- What is the evidence for and against ... -->
### FAQ
* Can models answer "who are you"?
* Does your feedback train the models?
* Why do models halucinate?
* Does a model always give the same answer to the same question?
* Can models "think"?
* How do models differ? Are there good and bad models?
* What is the difference between Open Source and Open Weight?
* Do models know about recent events - like news?
## Questions?