# CapsKG

This repository contains the code for the paper "CapsKG: Enabling Continual Knowledge Integration in Language Models for Automatic Knowledge Graph Completion"
https://link.springer.com/chapter/10.1007/978-3-031-47240-4_33

## Citation

If you use this code in your research, please cite the following paper:

```
Omeliyanenko, J., Zehe, A., Hotho, A., Schlör, D. (2023). CapsKG: Enabling Continual Knowledge Integration in Language Models for Automatic Knowledge Graph Completion. In: Payne, T.R., et al. The Semantic Web – ISWC 2023. ISWC 2023. Lecture Notes in Computer Science, vol 14265. Springer, Cham. https://doi.org/10.1007/978-3-031-47240-4_33
```

## Dependencies

Experiments were run using python 3.8

Pip dependencies are listed in the `requirements.txt` file.

## Data

The data used in the experiments are contained in the `dat` folder.
Used prompts are contained within the specific dataset folders in `templates.json`.

## Running the code

The code can be run using the `run.py` script.
Run arguments may be adjusted in the `config.py` file.

The following arguments are used for the specific models run in the paper:

- BERT: `--backbone bert  --baseline one_mlm`
- Adapter: `--backbone bert_adapter  --baseline one_mlm`
- BERT-CL: `--backbone bert  --baseline ncl_mlm`
- Adapter-CL: `--backbone bert_adapter  --baseline ncl_mlm`
- CapsKG: `--backbone bert_adapter  --baseline ctr_kg`

Available Datasets are:

- WN18: `--task wn18`
- YAGO3-10: `--task yago`
- FB15k: `--task fbl`

Example usage:

`python3 run.py --bert_model bert-base-uncased --baseline ctr_kg --backbone bert_adapter --task wn18 --scenario "til_classification" --use_predefine_args --ntasks 5 --num_train_epochs 1 --yaml_param_num 0 --output_dir "\my_output_directory" --auto_resume --reuse_data`
