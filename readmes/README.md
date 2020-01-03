
## Experiments:

1. Multitarget stratification (see more in [`iterstat`](https://github.com/trent-b/iterative-stratification)) vs [KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)

2. Ensemble of models <br>
__15 Dec 2019__: works even with worth dataset. <br>
__02 Jan 2020__: more folds improves stability over folds (scores +- same), decreased local CV and public LB

3. Ensemble of models for each field

4. FP16 - [nvidia apex](https://github.com/NVIDIA/apex), [catalyst](https://github.com/catalyst-team/catalyst/blob/master/examples/_tests_mnist_stages2/config_fp16_O1.yml)

5. Check metric differences when stratification was made with <b>MultitargetKFold</b>

6. Check metric differences when stratification was made with <b>GroupKFold</b> <br>
__02 Jan 2020__: decreased score gap between local CV and public LB and decrease STD beetween folds

7. Text Augmentations ([nlpaug](https://github.com/makcedward/nlpaug))

8. Use categories (`category` & `host`) features with transformer model. <br>
__31 Dec 2019__: inproves local CV (__0.38346 +- 0.00671017138380235__ -> __0.38754 +- 0.006791641922245308__) and decreased public LB (__0.384__ -> __0.381__) <br>
<div style="padding-left: 4em;"> <b>NOTE:</b> all assumptions was made compared to the `bert-base` trained on same data but without categorical features </div>

9. Use text statistics (word, tokens, special symbols, whitespaces and other counts) with transformer model. <br>
__02 Jan 2020__: decreased score gap between local CV (__0.38858 +- 0.011114207124217198__) and public LB (__0.388__) (based on `bert-base`)

