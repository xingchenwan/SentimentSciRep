# Code repository for *Sentiment correlation in financial news networks and associated market movements*

This repository contains the open-sourced codes for our paper at Scientific Reports 11. [[html]](https://www.nature.com/articles/s41598-021-82338-6#citeas) [[arxiv]](https://arxiv.org/abs/2011.06430)
## Requirements
```
(Anaconda) Python 3.6
matplotlib
pandas
numpy~=1.19.2
networkx
nltk~=3.4.5
scikit-learn~=0.23.2
scipy~=1.3.1
seaborn~=0.11.0
python-louvain~=0.13
xlrd~=1.2.0
tqdm~=4.54.1
```

You might additionally need Microsoft Excel to process market data.

## Instructions

### 1. Co-occurrence matrix

We include processed co-occurrence matrix available at ```occurrence_network``` under the root directory. You need to first
unpickle it into a networkx format. Note that this is the raw co-occurrence matrix with possible non-zero edge weight
between any pair of companies. To reconstruct the result shown in our paper, you need to follow the instructions in the
paper by pruning the graph by retaining the top-neighbour of each company only.

*Note: the raw articles used to compute the sentiment scores and the co-occurrence network are copyrighted materials of
Thomson Reuters and will not be made available in this repo. You need to consult their Term of Use [here](https://www.thomsonreuters.com/en/terms-of-use.html).*

### 2. Sentiment and market data results

You may consult ```main.py``` for the various routines to generate the results. We have included a dummy ```market_data.xlsx``` under
   ```./data``` for reference of the format of market data file expected. The repo also expects a ```full_data``` object (interface defined under ```source/all_classes.py```)
   containing the relevant article and sentiment information.

*Note: similar to the Reuters articles, we are also not releasing the market data which are proprietary. Data of comparable quality may be, for example, obtained from Yahoo Finance and you are encouraged to review their Terms of Use.* 


## Citation

If you find this code repository or the paper to be useful for your research, please consider citing:

Wan, X., Yang, J., Marinov, S., Calliess, JP., Zohren, S., and Dong X. Sentiment correlation in financial news networks and associated market movements. Sci Rep 11, 3062 (2021). https://doi.org/10.1038/s41598-021-82338-6

(Equal contribution between X Wan and J Yang)

Alternatively, in biblex format:

```
@article{wan2021sentiment,
  title={Sentiment correlation in financial news networks and associated market movements},
  author={Wan, Xingchen and Yang, Jie and Marinov, Slavi and Calliess, Jan-Peter and Zohren, Stefan and Dong, Xiaowen},
  journal={Scientific reports},
  volume={11},
  year={2021},
  publisher={Nature Publishing Group}
}
```
