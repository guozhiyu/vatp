# Attention Score is not All You Need for Token Importance Indicator  in  KV Cache Reduction: Value Also Matters 
Implementation of EMNLP 2024 main conference paper [Attention Score is not All You Need for Token Importance Indicator  in  KV Cache Reduction: Value Also Matters](https://arxiv.org/abs/2406.12335)

Zhiyu Guo, Hidetaka Kamigaito, Taro Watanabe

Nara Institute of Science and Technology

## Setup

Step 1: Create a new conda environment:

```
conda create -n vatp python=3.10
conda activate vatp
```



Step 2: Install relevant packages

```
pip install -r requirement.txt 
```

## Observations of attention matrices and value vector norms
You can use `visual.ipynb` to visualize the attention matrices and value vector norms. You can use ‘layer_id’ and ‘head_id’ to specify the layer and head you want to visualize.

## Evaluate different KV cache reduction methods in LongBench
You can run generation in LongBench tasks  using half of the KV cache budget by the following command:
```
python pred.py --model llama2-7b-chat-4k --sink_len 20 --save outctllam2/h2ovatp --heavy_ratio 0.25 --e --h2o --apval
```
This is H2O w/ VATP variant. `--h2o` means using the setting of H2O, otherwise Scissorhands.  `--sink_len` means the length of the attention sink tokens. `--h2o` indicates the seting of H2O. Enable `--apval` for integration of VATP. `--e` indicates using tasks `["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", 
            "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]`, otherwise for  tasks `["narrativeqa","musique","qmsum"]`.
Note that here we follow the code of [H2O](https://github.com/FMInference/H2O/blob/main/h2o_hf/run_text_generation.py), in the task evaluation, we mask the tokens to be pruned, they are not actually dropped in the memory, so in the evaluation process, we cannot observe actual memory reduction and inference speedup.

You can achieve the evaluation results using the corresponding metrics for each task by running

``` 
python eval.py --model llama2-7b-chat-4k --save outctllam2/h2ovatp
```


## 🙏 Acknowledgement
This repo is built upon the following projects:

* [H2O](https://github.com/FMInference/H2O)
* [Massive Activations](https://github.com/locuslab/massive-activations)
* [LongBench](https://github.com/THUDM/LongBench)

We thank the authors for their code.

## 📝 Citation
We kindly request that you cite our work if you utilize the code or reference our findings in your research:
<!-- Please cite our work if you use our code or discuss our findings in your own research: -->
```
@article{guo2024attention,
  title={Attention Score is not All You Need for Token Importance Indicator in KV Cache Reduction: Value Also Matters},
  author={Guo, Zhiyu and Kamigaito, Hidetaka and Watanabe, Taro},
  journal={arXiv preprint arXiv:2406.12335},
  year={2024}
}
