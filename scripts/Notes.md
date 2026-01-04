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

The decision to leave the custom implementation of tokenization to using the `Tokenizers` library was made as the custom implementation was too unoptimized to be scaled to larger numbers of `BPE merges` and/or larger corpus.

The biggest factors in it unoptimized nature were being single-threaded behaviour, and having multiple levels of nested loops for `pair frequency counting` after performing each merge.

The custom implementation took more than 1 and a half hours to finish on a corpus of **41.5 MB** with **10000 merges** on an **i7-6600u**, but the `Tokenizers` library based approach with the same specification takes almost 2 minutes, which makes re-iterating a little more comfortable.

### End-Of-Word and New-Line Marks

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
