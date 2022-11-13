# CSE 6250 - Big Data for Health -Final Project
[Final Paper](https://github.com/trantonnq/ClassifySleepStages/blob/47a9c3eb498bcb99d1856fa406195c63c0e87fd8/Final%20Paper%20-%20Identifying%20NREM%20Sleep%20Stages%20in%20Consumer%20Wearables.pdf)

[Presentation](https://www.youtube.com/watch?v=xS4Df2oXbMg)

## Introduction
In this project, we built a sleep stage classifier using Apple Watch health data gathered from 31 subjects. This is an expansion of the work by Walch et al [1] [2] that is available on PhysioNet10. While the original work only classifies sleep into 3 stages Wake/NREM/REM, we successfully developed more advanced classifications which classify sleep into more specific classes (4 stages - Wake/N1/N2/N3/REM and 5 stages - Wake/N1+N2/N3/REM) to bring more benefits to sleep studies.

## Method

### Data
- Input data: Apple Watch data collected during subject's sleep - Heart rate, Motion, Circadian, Clock, Polysonorgaphy (PSG)
- Output data: Sleep stages scored by professional sleep technicians based on PSG data.
- Prepocessed data to create using features to train ML models
- Used Monte Carlo cross-validation 



### Metrics
- Accuracy and one-versus-rest ROC curve

### ML algorithms 
- Logistic Regression
- Random Forest
- k-Nearest-Neighbors
- AdaBoost
- Neural Net

## Results and Discussion
[Final Paper](https://github.com/trantonnq/ClassifySleepStages/blob/47a9c3eb498bcb99d1856fa406195c63c0e87fd8/Final%20Paper%20-%20Identifying%20NREM%20Sleep%20Stages%20in%20Consumer%20Wearables.pdf)

## How to run the code

**To run ML classifiers:**
- Run ```analysis_runner.py```in \source\analysis
- In "analysis_runner.py" main function:
>- figures_mc_five_class() is for 5-sleep-stage classification
>- figures_mc_four_class() is for 4-sleep-stage classification
- If the code takes too long to run or run into Memory Error due to limited local hardware, comment out some classifiers in the "classifiers" variable in the above two functions to run it one by one. 

**Installation:**
- Make sure to install packages listed in requirements.txt. If running into error when executing the code, might need to downgrade some packages to match with the version listed in requirements.txt.  You can also use environment.yml to set up an Anaconda environment for this set of code.

## Reference
[1] Walch, O. (2019). Motion and heart rate from a wrist-worn wearable and labeled sleep from polysomnography (version 1.0.0). PhysioNet. https://doi.org/10.13026/hmhs-py35.

[2] Olivia Walch, Yitong Huang, Daniel Forger, Cathy Goldstein, Sleep stage prediction with raw acceleration and photoplethysmography heart rate data derived from a consumer wearable device, Sleep, Volume 42, Issue 12, December 2019, zsz180, https://doi.org/10.1093/sleep/zsz180

[3] Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.

The original work's description is below:

### sleep_classifiers

This code uses scikit-learn to classify sleep based on acceleration and photoplethymography-derived heart rate from the Apple Watch. The paper associated with the work is available [here](https://academic.oup.com/sleep/article/42/12/zsz180/5549536).

### Getting Started

This code uses Python 3.7.

### Data

Data collected using the Apple Watch is available on PhysioNet: [link](https://alpha.physionet.org/content/sleep-accel/1.0.0/)

The MESA dataset is available for download at the [National Sleep Research Resource](https://sleepdata.org). You will have to request access from NSRR for the data.

### Features + figures

All raw data are cleaned and features are generated in ```preprocessing_runner.py.```

The file ```analysis_runner.py``` can be used to generate figures showing classifier performance.  You can comment and uncomment the figures you want to run. 

### Notes
- In the blue motion-only classifier performance lines in Figures 4 and 8 in [the paper](https://academic.oup.com/sleep/article/42/12/zsz180/5549536), labels for REM and NREM sleep are switched. NREM sleep is the dashed line and REM is the dotted line.
- The subset of the MESA dataset used for comparison in the paper are the first 188 subjects with valid data, in order of increasing Subject ID.

