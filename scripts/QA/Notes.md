# Notes

## Analyze Long Sentences - `sindhi_wiki_articles_cleaned.txt`
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
