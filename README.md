# MehranLM

![Status](https://img.shields.io/badge/Status-In_Progress-yellow)

**MehranLM** is an experimental attempt to build a **Small Language Model (SLM) for Sindhi**. 

To create a home-grown Sindhi language model that developers, students, and eventually the wider public can use, improve, and enjoy.

> <p align="left">جيو ڪريس! </p>

---

## Why MehranLM?

- **Lack of Sindhi support** - General or multilingual LLMs  perform poorly on Sindhi due to lack of data.
- **Local-first approach** - MehranLM is focused exclusively in the Sindhi language.
- **Community-driven** - Developers, students, and contributors are welcome to join hands in building a Sindhi language model.

---

## Audience

- **Students and Developers** - who want to experiment with language models and support Sindhi NLP.
- **Researchers** - (future scope) - to analyze, fine-tune, and evaluate a dedicated Sindhi Language Model.
- **General Public Use** - (future scope) - applications like text-completion, translation, or chatbots.

---

## Roadmap

- [ ] Collect and clean Sindhi text dataset *(in-progress)*
  - [x] Download and Clean Wikidumps
  - [ ] Typing some stuff in Sindhi *(in-progress)*
  - [ ] OCR Public-Domain Books
  - [ ] Type folklore and old poetry
- [x] Tokenization
  - [x] Basic BPE Tokenization
  - [ ] Further Improvements
- [ ] Train on RNN
- [ ] Train on LSTM
- [ ] Train on Transformers
- [ ] Benchmarks

---

## Contribution

This project cannot grow alone. Contributions are welcome. You can help us in many ways.
- Dataset collection and cleaning
- Model design and training
- Evaluation benchmarks
- Documentation and tutorials

### Guide

- Clone this repository (for Devs and Students)
    ```
    git clone https://github.com/IshaqJunejo/MehranLM.git
    ```
- Open an issue on this repository for any suggestions
- Or reach out via [email](mailto:ishaque.junejo.dev@gmail.com), [LinkedIn](https://www.linkedin.com/in/ishaque-junejo/), or anywhere else you can find.

---

## License

This repository contains 2 different licenses.

- **Pipeline (`pipeline/`)**: Licensed under the MIT License - see [LICENSE](LICENSE)
- **Corpus (`Corpus/`)**: License under CC-BY-SA-4.0, as portions of this corpus is derived from Wikimedia content (Wikipedia dumps) - see [LICENSE](Corpus/LICENSE)
