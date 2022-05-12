# KoBERTSEG: Local Context Based Topic Segmentation Using KoBERT

This is an official code for **\<KoBERTSEG: Local Context Based Topic Segmentation Using KoBERT, JKIIE, 2022>.**<br>
<i>Codes for building backbone BERT architecture refers to [Presumm](https://github.com/nlpyang/PreSumm).</i>

## KoBERTSEG

### Highlights
KoBERTSEG is a KoBERT based model for topic segmentation, which has its main objective in splitting a document into sub-documents each having only one single topic.
<p align="center">
<img src="misc/kobertseg_structure.png " width="600"> 
</p>

Our main contributions are as follows:

* KoBERTSEG uses the structure used for document summarization, BERTSUM, which is able to embed per-sentence representation for give document input, and as a result can calculate relationships between each sentence which is critical for topic segmentation.

* By taking advantage of long-sequence understanding ability of KoBERTSEG, it reduces False Positive error by margin compared to other methods, which is critical for down-stream tasks such as document summarization, document classification, etc.

* Not only introducing the strcuture to be used for topic segmentation, we introduce a novel way of using a huge Korean dataset, which was originally constructed for summarization, for topic segmentation. By exploiting such a way of making dataset and training on it, we could train a model that can be generalized for news articles covering a wide range.
<br><br>

### Evaluation

| Model | Precision | Recall | F-1 Score | $p_k$ | WindowDiff |
| :------------: | :------------: | :------------: | :------------: | :------------: | :------------: |
| `Random` | 6.40 | 6.30 | 6.34 | 0.5077 | 0.4877 |
| `Even` | 7.76 | 7.75 | 7.75 | 0.4118 | 0.4118 |
| `LSTM`</br>(Koshorek et al., 2018) | 67.66 | 89.80 | 76.25 | 0.1526 | 0.1531 |
| `BTS`</br>(Jeon et al., 2019) | 90.02 | <u>**98.90**</u> | 93.65 | 0.0355 | 0.0457 |
| `KoBERTSEG`</br>(window=3)| 95.69 | 98.50 | 96.79 | 0.0193 | 0.0233 |
| `KoBERTSEG-SUM`</br>(window=3)| <u>**97.38**</u> | 98.78 | <u>**97.86**</u> | <u>**0.0136**</u> | <u>**0.0157**</u> |
<br>

## Usage

### Requirements
Linux, Python>=3.8, PyTorch>=1.7.1

We recommend you to use Anaconda to create a conda environment:
```bash
conda create -n kobertseg python=3.7 pip
conda install pytorch=1.7.1 cudatoolkit=9.2 -c pytorch
```

### Dataset(bfly)
You can download bfly-soft article dataset [here](https://drive.google.com/file/d/1kZ3_xIzbXNalci7RuS2YeR5ksRW23uNr/view?usp=sharing) for training/evaluation; unzip and put the files(train.jsonl, dev.jsonl, test.jsonl) under directory `dataset/bfly`

s### Simple Inference
With trained [model](https://drive.google.com/file/d/1ZIfZM_LsPn0MWh12nOkyx6alTY1viw5I/view?usp=sharing) at `models/best_model.pth` and demo text file at `./simple_inference/demo.story`, implement simple inference using the command below;
```bash
python simple_inference.py --test_from models/best_model.pth
```
This command will produce `inference_result.txt` file under the directory `./simple_inference`. It produces logit scores for topic segmentation between every two sentences.
