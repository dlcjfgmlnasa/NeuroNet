# NeuroNet: A Novel Hybrid Self-Supervised Learning Framework for Sleep Stage Classification Using Single-Channel EEG

>**Cheol-Hui Lee, Hakseung Kim, Hyun-jee Han, Min-Kyung Jung, Byung C. Yoon and Dong-Joo Kim**

[[`Paper`](https://arxiv.org/abs/2404.17585)] [[`Paper with Code`](https://paperswithcode.com/paper/neuronet-a-novel-hybrid-self-supervised)] [[`BibTeX`](#license-and-citation)] 

**Full code coming soon^^**

![neuronet structure](https://github.com/dlcjfgmlnasa/NeuroNet/blob/main/figures/model_structure.jpg)

## Introduction 🔥
The classification of sleep stages is a pivotal aspect of diagnosing sleep disorders and evaluating sleep quality. However, the conventional manual scoring process, conducted by clinicians, is time-consuming and prone to human bias. Recent advancements in deep learning have substantially propelled the automation of sleep stage classification. Nevertheless, challenges persist, including the need for large datasets with labels and the inherent biases in human-generated annotations. This paper introduces NeuroNet, a self-supervised learning (SSL) framework designed to effectively harness unlabeled single-channel sleep electroencephalogram (EEG) signals by integrating contrastive learning tasks and masked prediction tasks. NeuroNet demonstrates superior performance over existing SSL methodologies through extensive experimentation conducted across three polysomnography (PSG) datasets. Additionally, this study proposes a Mamba-based temporal context module to capture the relationships among diverse EEG epochs. Combining NeuroNet with the Mamba-based temporal context module has demonstrated the capability to achieve, or even surpass, the performance of the latest supervised learning methodologies, even with a limited amount of labeled data. This study is expected to establish a new benchmark in sleep stage classification, promising to guide future research and applications in the field of sleep analysis.

## Highlights ✨

## Main Result 🥇

<details>
<summary> <i> <b> Performance of Sleep-EDFX across various self-supervised learning and supervised learning </b> </i> </summary>
<p align="center"> <img src="https://github.com/dlcjfgmlnasa/NeuroNet/blob/main/figures/overview.jpg" alt="image" width="80%" height="auto"> </p>
</details>

<details>
<summary> <i> <b>  The output hypnograms across five sleep stages. </b> </i> </summary>
<p align="center"> <img src="https://github.com/dlcjfgmlnasa/NeuroNet/blob/main/figures/hypnogram.jpg" alt="image"> </p>
The first, second, and third columns correspond to #sc4031e0, #shhs1-204928, and #subject-53 within Sleep-EDFX, SHHS, and ISRUC, respectively. (A) is manually scored by a sleep expert. (B) and (C) respectively represent NeuroNet-B and NeuroNet-T. The first row for both (B) and (C) displays the results for NeuroNet+TCM, while the second row shows the results for NeuroNet. The errors are marked by the red dots.
</details>


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
