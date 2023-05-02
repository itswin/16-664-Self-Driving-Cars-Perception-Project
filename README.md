# 16-664-Self-Driving-Cars-Perception-Project

This is the repository for the 16-664 Self-Driving Cars Perception Project. The kaggle link for the competition is [here](https://www.kaggle.com/competitions/16664-spring-2023-task-1-image-classification).

The `test` and `trainval` folders should be in the same directory as `train.py`.

To train, run `python train.py --train`, optionally specifying a number of epochs with `--epochs`. To start training from the latest checkpoint, run `python train.py --train --checkpoint`.

To test, run `python train.py --test`. This will test the model on the test set and output the results to `test_labels.csv`.
