# Notes

## Quality Assurance

### Analyze Long Sentences - `sindhi_wiki_articles_cleaned.txt`
* **1st Jan, 2026**

The python script `analyze_long_sentence.py` reads the corpus file `sindhi_wiki_articles_cleaned.txt` as it has a high number of large token-length sentences (as per the analysis).
The threshold for a **long sentence** is 75 tokens, there were almost 5500 sentences whose token-length exceeded this threshold. 
We collect all the sentences that exceed this threshold, and randomly sample 25 of those sentences and write them to `sentence_longer_than_75_tokens.txt` file.

#### Conclusion / Hypothesis

My hypothesis based on what I read from the output text file, is that the problem not only lies in the corpus-cleaning processing (or the wikidumps itself), but it also exists in the BPE-training.
As the number of merges for BPE were limited due to compute limitation, and because of that some of the tokens are a little small.
Which means, that the sentence length is also (at least somewhat) inflated by just having smaller tokens in general.

#### Next steps

- Migrate the BPE-training from custom python logic to some optimized library based solution, and experiement with a bigger number of merges
- Those sentences that still exceed the threshold will be analyzed again.
- If they seem to have broken semantics, we are getting rid of them.

## Tokenization

### Migration from Custom Tokenization to Tokenizers Library

* **3rd Jan, 2026**

The decision to leave the custom implementation of tokenization to using the `Tokenizers` library was made as the custom implementation was too unoptimized to be scaled to larger numbers of `BPE merges` and/or larger corpus.

The biggest factors in it unoptimized nature were being single-threaded behaviour, and having multiple levels of nested loops for `pair frequency counting` after performing each merge.

The custom implementation took more than 1 and a half hours to finish on a corpus of **41.5 MB** with **10000 merges** on an **i7-6600u**, but the `Tokenizers` library based approach with the same specification takes almost 2 minutes, which makes re-iterating a little more comfortable.

### End-Of-Word and New-Line Marks

* **5th Jan, 2026**

Previously, I was using `<nl>` and `<w/>` as markers for **new line** and **end of word**.

But when using the `Tokenizers` library for tokenization, I couldn't find a configuration for adding these as *markers* instead of *special tokens*, because the *special tokens* were not being added with the result from the decoding, which means they couldn't be swapped for handling **new lines and spaces**.

And using these markers without adding them to the *special tokens* made these markers being split in the tokenization aggressively, which inflated the token count.
```
<
nl
>
```

Instead, I decided to use `NL` and `EW` as the markers for **new lines** and **end of word**.

They work good enough because they are small enough to be guaranteed to be merged, and they aren't going to mess with the rest of the corpus because the entire corpus is now exclusively in the Sindhi Language (written in Arabic script).

Using single chars of Latin script could be an even better idea, but it is left out for the sake of ease of understanding.

### Special Tokens

* **5th Jan, 2026**

Added more special tokens to the tokenizer, because the plan for the language model is to be able to **extend** or **finetune** for different tasks.

Previously there were only 2 special tokens, `<UNK>` for *Unknown Tokens* and `<PAD>` for *Padding*.

Now, there are `<BOS>` for *Beginning of Sequence*, `<EOS>` for *End of Sequence*, `<SEP>` for *Sequence Seperator*, and `<MASK>` for *Masking token*, the last one will likely be used for predicting missing/masked tokens from a given sequence, in tasks like **OCR Correction**.

### Reducing Excessive Cleanup of Corpus

* **23rd March, 2026**

When trying to hand-pick a **subcorpus**, which can be used to train a char-based LSTM, which can then be used to analyze the perplexity of the entire corpus, I came across a problem. I realized that I had cleaned the corpus, a little too aggressively.
Specially, when the wikipedia articles had to mention a year, there were no numbers, only a 'ﻉ' (representing the year in AD calendar).

So I have decided to not remove *Numbers* and *Latin Characters*, but only allows lines with 75% of characters being in the Arabic Unicode ranges.

## Model

### Calculating the Perplexity Scores of the Corpus

* **9th June, 2026**

Having trained a char-based LSTM on a hand-picked **subcorpus** of around 260-270 KBs of text, I wanted to use that Language Model as loose benchmark for analyzing the larger corpus I have on hand.

So, I performed the analysis on each file individually, broke that file into chunks of 512 characters, and calculated the perplexity by comparing the prediction (by charLM) and the actual character, for every **8th** character in the chunk (stepping 8 characters for speeding purposes). Stored the perplexity score of every chunk, alongside with *mean*, *median*, and *standard deviation* of the chunk perplexity scores in a JSON file for each corpus file.

* `self_typed_corpus_00_cleaned.txt` has an average perplexity score of **48.74**.
* `sindhi_legal_dataset_cleaned.txt` has an average perplexity score of **11.40**.
* `sindhi_wiki_articles_cleaned.txt` has an average perplexity score of **11.68**.
* `sindhi_wiki_articles_index_cleaned.txt` has an average perplexity score of **24.27**.

`self_typed_corpus_00_cleaned.txt` and `sindhi_wiki_articles_index_cleaned.txt` has a really high perplexity, this is likely due the high density short, list based content in these 2 files.

The other 2 files, `sindhi_legal_dataset_cleaned.txt` and `sindhi_wiki_articles_cleaned.txt` does have a relatively lower perplexity, but still it is not low enough to be considered a "good enough" score. It is likely due to the limitation of the hand-picked subcorpus on which the charLM was trained. 