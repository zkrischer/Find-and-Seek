<p align="center">
<h1 align="center">AGILE3D: Attention Guided Interactive Multi-object 3D Segmentation</h1>
<p align="center">
<a href="https://n.ethz.ch/~yuayue/"><strong>Yuanwen Yue</strong></a>
,
<a href="https://www.vision.rwth-aachen.de/person/218/"><strong>Sabarinath Mahadevan</strong></a>
,
<a href="https://jonasschult.github.io/"><strong>Jonas Schult</strong></a>
,
<a href="https://francisengelmann.github.io/"><strong>Francis Engelmann</strong></a>
<br>
<a href="https://www.vision.rwth-aachen.de/person/1/"><strong>Bastian Leibe</strong></a>
, 
<a href="https://igp.ethz.ch/personen/person-detail.html?persid=143986"><strong>Konrad Schindler</strong></a>
,
<a href="https://theodorakontogianni.github.io/"><strong>Theodora Kontogianni</strong></a>
</p>
<h2 align="center">ICLR 2024</h2>
<h3 align="center"><a href="https://arxiv.org/abs/2306.00977">Paper</a> | <a href="https://ywyue.github.io/AGILE3D/">Project Webpage</a></h3>
</p>
<p align="center">
<img src="./imgs/teaser.gif" width="500"/>
</p>
<p align="center">
<strong>AGILE3D</strong> supports interactive multi-object 3D segmentation, where a user collaborates with a deep learning model to segment multiple 3D objects simultaneously, by providing interactive clicks.
</p>

## News :loudspeaker:

- [2024/02/05] Benchmark data, training and evaluation code were released.
- [2024/01/19] Our interactive segmentation tool was released. Try your own scans! :smiley:
- [2024/01/16] AGILE3D was accepted to ICLR 2024 :tada:


<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation-hammer">Installation</a>
    </li>
    <li>
      <a href="#interactive-tool-video_game">Interactive Tool</a>
    </li>
    <li>
      <a href="#benchmark-setup-dart">Benchmark Setup</a>
    </li>
    <li>
      <a href="#training-rocket">Training</a>
    </li>
    <li>
      <a href="#evaluation-chart_with_upwards_trend">Evaluation</a>
    </li>
    <li>
      <a href="#citation-mortar_board">Citation</a>
    </li>
    <li>
      <a href="#acknowledgment-pray">Acknowledgment</a>
    </li>
  </ol>
</details>

## Installation :hammer:

Foe training and evaluation, please follow the [installation.md](https://github.com/ywyue/AGILE3D/tree/main/installation.md) to set up the environments.

## Interactive Tool :video_game:

Please follow [this instruction](https://github.com/ywyue/AGILE3D/tree/main/demo.md) to play with the interactive tool yourself.  **It also works without GPU.**

<p align="center">
<img src="./imgs/demo.gif" width="75%" />
</p>

We present an **interactive** tool that allows users to segment/annotate **multiple 3D objects** together, in an **open-world** setting. Although the model was only trained on ScanNet training set, it can also segment unseen datasets like S3DIS, ARKitScenes, and even outdoor scans like KITTI-360. Please check the [project page](https://ywyue.github.io/AGILE3D/) for more demos. Also try your own scans :smiley:

## Benchmark Setup :dart:

We conduct evaluation in both *interactive single-object 3D segmentation* and *interactive multi-object 3D segmentation*. For the former, we adopt the protocol from [InterObject3D](https://github.com/theodorakontogianni/InterObject3D). For the latter, we propose our own setup since there was no prior work.

Our quantitative evaluation involves the following datasets: ScanNet (inc. ScanNet40 and ScanNet20), S3DIS and KITTI-360. We provide the processed data in the required format for both benchmarks. You can download the data from [here](https://drive.google.com/file/d/1cqWgVlwYHRPeWJB-YJdz-mS5njbH4SnG/view?usp=sharing). Please unzip them to the `data` folder.

If you want to learn more about the benchmark setup, explanations for the processed data, and data processing scripts, see the 
[benchmark document](https://github.com/ywyue/AGILE3D/tree/main/benchmark/README.md).


## Training :rocket:

We train a single model in multi-object setup on ScanNet40 training set. Once trained, we evaluate the model on both multi-object and single-object setups on ScanNet40, S3DIS, KITTI-360. 

The command for training AGILE3D with iterative training on ScanNet40 is as follows:

```shell
./scripts/train_multi_scannet40.sh
```

> Note: in the paper we also conducted one experiment where we train AGILE3D on ScanNet20 and evaluate the model on ScanNet40 (1st row in Tab. 1). Instructions for this setup will come later.

## Evaluation :chart_with_upwards_trend:

Download the pretrained [model](https://polybox.ethz.ch/index.php/s/RnB1o8X7g1jL0lM) and move it to the `weights` folder.

### Evaluation on interactive multi-object 3D segmentation:

- ScanNet40:
```shell
./scripts/eval_multi_scannet40.sh
```
- S3DIS:
```shell
./scripts/eval_multi_s3dis.sh
```
- KITTI-360:
```shell
./scripts/eval_multi_kitti360.sh
```

### Evaluation on interactive single-object 3D segmentation:

- ScanNet40:
```shell
./scripts/eval_single_scannet40.sh
```
- S3DIS:
```shell
./scripts/eval_single_s3dis.sh
```
- KITTI-360:
```shell
./scripts/eval_single_kitti360.sh
```


## Citation :mortar_board:

If you find our code or paper useful, please cite:

```
@inproceedings{yue2023agile3d,
  title     = {{AGILE3D: Attention Guided Interactive Multi-object 3D Segmentation}},
  author    = {Yue, Yuanwen and Mahadevan, Sabarinath and Schult, Jonas and Engelmann, Francis and Leibe, Bastian and Schindler, Konrad and Kontogianni, Theodora},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2024}
}
```

## Acknowledgment :pray:

**We sincerely thank all volunteers who participated in our user study!** Francis Engelmann and Theodora Kontogianni are postdoctoral research fellows at the ETH AI Center. This project is partially funded by the ETH Career Seed Award - Towards Open-World 3D Scene Understanding,
NeuroSys-D (03ZU1106DA) and BMBF projects 6GEM (16KISK036K).

Parts of our code are built on top of [Mask3D](https://github.com/JonasSchult/Mask3D) and [InterObject3D](https://github.com/theodorakontogianni/InterObject3D). We also thank Anne Marx for the help in the initial version of the GUI.
