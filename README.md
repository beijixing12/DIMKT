# DIMKT

Source code and data set for our paper (recently accepted in SIGIR2022): Assessing Student's Dynamic Knowledge State by Exploring the Question Difficulty Effect.

The code is the implementation of DIMKT model, and the data set is the public data set [ASSIST2012-2013](https://sites.google.com/site/assistmentsdata/home/2012-13-school-data-withaffect).



## Dependencies

- python >= 3.7
- pytorch >= 1.12
- numpy
- tqdm
- utils
- pandas
- sklearn


## Usage

First, download your raw CSV files (e.g., `2012-2013-data-with-predictions-4-final.csv`) or NPZ files (e.g., `assist09.npz`, `OLI.npz`) into the `data/` folder.

Run `prepare_dataset.py` to perform both `data_pre.py` and `data_save.py` steps in one command. The script writes mapping/difficulty files (`user2id`, `problem2id`, `skill2id`, `difficult2id`, `sdifficult2id`, `nones.npy`, `nonesk.npy`) **and** the sliced splits (`train0.npy`, `valid0.npy`, `test.npy`) directly into `data/`. 低频题目/技能会按原始规则保留默认难度，若过滤后切片为空将自动回退为不过滤以保证生成可训练序列。

```bash
# One-step preprocessing and splitting
python prepare_dataset.py data/2012-2013-data-with-predictions-4-final.csv --seq_len 100
python prepare_dataset.py data/assist09.npz --seq_len 100
python prepare_dataset.py --input_csv data/OLI.npz --seq_len 150  # optional flag form
```

The NPZ reader accepts the following column names (aliases are supported):
> - `user_id` (`uid`, `user`, `student_id`, `stu_id`)
> - `problem_id` (`item_id`, `pid`, `question_id`, `item`)
> - `correct` (`label`, `is_correct`, `score`, `y`, `attempt`)
> - `skill_id` (`skill`, `kc`, `concept_id`)
> - `end_time` (`timestamp`, `time`, `ts`, `unix_time`, `response_time`, `first_response_time`)
>   - If both `first_response_time` and `response_time` exist, they are summed to form `end_time`; if no end-time-like column exists or cannot be parsed, the script synthesizes a deterministic ordering so slicing can proceed.

### Train / Evaluate

After running `prepare_dataset.py`, the `data/` folder will contain `train0.npy`, `valid0.npy`, and `test.npy`. Run training with the appropriate fold index:

```bash
python train.py 0  # or 1,2,3,4 depending on the desired validation fold
python test.py <model_checkpoint_name>
```


