# Corpus

## Overview

This directory contains the corpus used in the project. It includes cleaned and some raw versions of the data.

## Directory Structure

- `**Cleaned/**`: This directory contains the processed files.
- `**Raw/**`: This directory contains the raw, unprocessed corpus files.
  - `**Large/**`, contains raw files that are too large for GitHub's file size limit; these have been placed in `.gitignore`.
- `**Private/**`: This has been placed in `.gitignore`
  - `**Cleaned/**`, this directory contains cleaned versions of private corpus.
  - `***Raw/*`, this directory contains raw versions of private corpus.
- `***subcorpus-for-charLM***`: This directory contains handpicked contents from the file `Cleaned/` directory. This handpicked content was selected on the basis of coherence and used to train the char-based LSTM.
- `***Testing-data***`: This directory contains held-out datasets (not used in training), so they can be used as testing datasets.

## Acknowledgements

A considerable portion of this corpus is derived from **Wikimedia Content (Wikipedia Dumps)**, that is why this project's corpus is licensed under **CC-BY-SA-4.0**, see [LICENSE](LICENSE) for more information.

To [Danish Mahdi](https://www.linkedin.com/in/danish-mahdi-571265231/) for making the [Sindhi Legal Dataset](https://huggingface.co/datasets/DanishMahdi/Sindhi_Legal) available under the license **CC-BY-SA-4.0**.

To [Fahad Maqsood Qazi](https://www.linkedin.com/in/fahad-maqsood-qazi) for making the [Sindhi Misspelled Sentences Dataset](https://huggingface.co/datasets/fahadqazi/Sindhi-Misspelled-Sentences) available on HuggingFace.

To [Ali Nawaz Mahar](https://www.linkedin.com/in/ali-nawaz-786w) for making the [Sindhi Sentiment Dataset](https://huggingface.co/datasets/alinawazmahar/Sindhi_Sentiment_dataset) available under the license **CC-BY-4.0**.
