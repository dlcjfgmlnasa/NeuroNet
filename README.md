# NeuroNet: A Novel Hybrid Self-Supervised Learning Framework for Sleep Stage Classification Using Single-Channel EEG

>**Cheol-Hui Lee, Hakseung Kim, Hyun-jee Han, Min-Kyung Jung, Byung C. Yoon and Dong-Joo Kim**

[[`Paper`](https://arxiv.org/abs/2404.17585)] [[`BibTeX`](#license-and-citation)]

![neuronet structure](https://github.com/dlcjfgmlnasa/NeuroNet/blob/main/figures/model_structure.jpg)

## Introduction 🔥
The classification of sleep stages is a pivotal aspect of diagnosing sleep disorders and evaluating sleep quality. However, the conventional manual scoring process, conducted by clinicians, is time-consuming and prone to human bias. Recent advancements in deep learning have substantially propelled the automation of sleep stage classification. Nevertheless, challenges persist, including the need for large datasets with labels and the inherent biases in human-generated annotations. This paper introduces NeuroNet, a self-supervised learning (SSL) framework designed to effectively harness unlabeled single-channel sleep electroencephalogram (EEG) signals by integrating contrastive learning tasks and masked prediction tasks. NeuroNet demonstrates superior performance over existing SSL methodologies through extensive experimentation conducted across three polysomnography (PSG) datasets. Additionally, this study proposes a Mamba-based temporal context module to capture the relationships among diverse EEG epochs. Combining NeuroNet with the Mamba-based temporal context module has demonstrated the capability to achieve, or even surpass, the performance of the latest supervised learning methodologies, even with a limited amount of labeled data. This study is expected to establish a new benchmark in sleep stage classification, promising to guide future research and applications in the field of sleep analysis.

## Highlights ✨

## Main Result 🥇

- *Performance of Sleep-EDFX across various self-supervised learning and supervised learning*
<p align="center">
      <img src="https://github.com/dlcjfgmlnasa/NeuroNet/blob/main/figures/overview.jpg" alt="image" width="60%" height="auto">
</p>

## License and Citation 📰
The software is licensed under the Apache License 2.0. Please cite the following paper if you have used this code:
```
@misc{lee2024neuronet,
      title={NeuroNet: A Novel Hybrid Self-Supervised Learning Framework for Sleep Stage Classification Using Single-Channel EEG}, 
      author={Cheol-Hui Lee and Hakseung Kim and Hyun-jee Han and Min-Kyung Jung and Byung C. Yoon and Dong-Joo Kim},
      year={2024},
      eprint={2404.17585},
      archivePrefix={arXiv},
      primaryClass={cs.HC}
}
```
