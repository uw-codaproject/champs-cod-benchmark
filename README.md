# A Multimodal Benchmark for Evaluating Cause-of-Death Inference Using Child Health and Mortality Data

Code for our NeurIPS 2026 paper (Evaluations and Datasets Track).

## Authors

Junhe Yang<sup>1\*</sup>, Soumyakanti Pan<sup>1\*</sup>, Hyun Seung Lim<sup>2</sup>,
Yue Chu<sup>3</sup>, Yuting Guo<sup>4</sup>, Nishtha Agarwal<sup>1</sup>,
Varun Babbar<sup>5</sup>, Gaurav Rajesh Parikh<sup>5</sup>, Yiqun T. Chen<sup>6</sup>,
Chris A. Rees<sup>4</sup>, Ziyaad Dangor<sup>7</sup>, Sanjay G. Lala<sup>7</sup>,
Zehang Richard Li<sup>8</sup>, Samuel J. Clark<sup>3</sup>, Zhenke Wu<sup>9</sup>,
Abhirup Datta<sup>6</sup>, Li Liu<sup>6</sup>, Cynthia Rudin<sup>5</sup>,
Samuel V. Scarpino<sup>2</sup>, Benjamin M. Gyori<sup>2</sup>, Tyler H. McCormick<sup>1</sup>

<sup>1</sup>University of Washington, <sup>2</sup>Northeastern University,
<sup>3</sup>Ohio State University, <sup>4</sup>Emory University,
<sup>5</sup>Duke University, <sup>6</sup>Johns Hopkins University,
<sup>7</sup>University of the Witwatersrand, <sup>8</sup>University of California, Santa Cruz,
<sup>9</sup>University of Michigan

<sup>\*</sup>Joint first authors

## About the paper

Accurately attributing causes of death is vital for global health, yet only 8% of reported deaths in low-income countries have a documented cause. To assign causes to these unlabeled deaths at scale, practitioners traditionally rely on verbal autopsy, using supervised statistical models to classify based on structured survey data. However, modern mortality surveillance increasingly collects rich, unstructured multimodal data, such as free-text caregiver narratives and postmortem diagnostics, which traditional supervised statistical models struggle to seamlessly integrate. In this paper, we present a comprehensive, multimodal benchmark for cause-of-death classification using data from the Child Health and Mortality Prevention Surveillance (CHAMPS) network, a unique surveillance platform spanning nine countries across South Asia and Sub-Saharan Africa. Using this dataset, we introduce an evaluation framework designed to rigorously assess diagnostic reasoning, moving beyond traditional metrics that fail to capture complex clinical realities. We demonstrate the utility of this benchmark by evaluating pretrained large language models used through prompting, without task-specific fine-tuning, against supervised baselines across progressively richer evidence configurations. Our results reveal distinct differences in how these modeling approaches use unstructured medical evidence. This benchmark provides a rigorously defined resource for assessing clinical reasoning in next-generation mortality surveillance.

## Repository layout

| Folder | Contents |
|--------|----------|
| [`LLM-analysis/`](LLM-analysis/) | The LLM experiments: gpt-oss-120b and gpt-oss-20b |
| [`statsML-analysis/`](statsML-analysis/) | The supervised and VA baselines (LightGBM, Elastic Net, InSilicoVA), the fixed train/test split, and the bootstrap analysis and figures for all methods |

Each folder has its own README with setup and reproduction steps.

## Data

The CHAMPS Level-2 de-identified data are available under a data use agreement
and are not included in this repository. Access is granted by the CHAMPS network
on application: https://champshealth.org/data/.

## Citation

```bibtex
@inproceedings{yang2026multimodal,
  title     = {A Multimodal Benchmark for Evaluating Cause-of-Death Inference
               Using Child Health and Mortality Data},
  author    = {Yang, Junhe and Pan, Soumyakanti and Lim, Hyun Seung and
               Chu, Yue and Guo, Yuting and Agarwal, Nishtha and
               Babbar, Varun and Parikh, Gaurav Rajesh and Chen, Yiqun T. and
               Rees, Chris A. and Dangor, Ziyaad and Lala, Sanjay G. and
               Li, Zehang Richard and Clark, Samuel J. and Wu, Zhenke and
               Datta, Abhirup and Liu, Li and Rudin, Cynthia and
               Scarpino, Samuel V. and Gyori, Benjamin M. and
               McCormick, Tyler H.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS),
               Evaluations and Datasets Track},
  year      = {2026}
}
```

## License

MIT License, copyright University of Washington (`LICENSE`). It covers this
code only, not the models it calls or the data it reads. The bundled InSilicoVA
R package (`statsML-analysis/insilicova/InSilicoVAx_1.4.2.tar.gz`) keeps its own
GPL-2 license.
