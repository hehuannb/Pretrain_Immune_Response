Time Series Foundation Model Data Curation

## Sources

This corhort contains 19 multi-variate time series regression datasets, 112 univaraite time Series classification datasets, and 26 multivariate time series classification dataset. 

More descriptions of the datasets can be found at our Raindrop paper: https://openreview.net/pdf?id=Kwm8I7dU-l5 and TF-C submission https://github.com/mims-harvard/TFC-pretraining

## Raw Datasets and descriptions

1. P19 denotes PhysioNet Sepsis Early Prediction Challenge 2019. Raw data of P19 can be found at https://physionet.org/content/challenge-2019/1.0.0/

2. P12 denotes PhysioNet Mortality Prediction Challenge 2012. aw data of P12 can be found at https://physionet.org/content/challenge-2012/1.0.0/

3. PAM is a subset of PAMAP2 Physical Activity Monitoring. Raw data of PAM can be found at http://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring

4. SleepEEG is from Sleep-EDF Database. Raw data of SleepEEG can be found at https://www.physionet.org/content/sleep-edfx/1.0.0/

5. Epilepsy is from http://hdl.handle.net/10230/42894

6. FD-A is a fault diagnosis dataset. It is gathered from an electromechanical drive system that monitors the condition of rolling bearings and detects damages in them. The raw data can be found at https://mb.uni-paderborn.de/en/kat/main-research/datacenter/bearing-datacenter/data-sets-and-download

7. FD-B is from the same cohort of FD-A. It measures a different rolling bearing. The raw data can be also found at https://mb.uni-paderborn.de/en/kat/main-research/datacenter/bearing-datacenter/data-sets-and-download

8. HAR is from Human Activity Recogntion Using Smartphones Data Set. The raw data can be found at https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones

9. Gesture is a hand-gesture recognition dataset. The raw data can be found at http://www.timeseriesclassification.com/description.php?Dataset=UWaveGestureLibrary

10. ECG is from the 2017 PhysioNet Challenge that focuses on ECG recording classification. The raw data can be found at https://physionet.org/content/challenge-2017/1.0.0/

11. EMG is a simple electromyogram dataset measure the status of three subjects.The raw dataset can be found at https://physionet.org/content/emgdb/1.0.0/


## Processed datasets

The well-processed datasets can be founded in this folder. Each dataset has a separate folder. The .pt files are saved in python format and can be easily read by python. 

The processed datasets are also released publicly in FigShare:

1. P19** (PhysioNet Sepsis Early Prediction Challenge 2019) https://doi.org/10.6084/m9.figshare.19514338.v1

2. **P12** (PhysioNet Mortality Prediction Challenge 2012) https://doi.org/10.6084/m9.figshare.19514341.v1

3. **PAM** (PAMAP2 Physical Activity Monitoring) https://doi.org/10.6084/m9.figshare.19514347.v1

4. **SleepEEG**: https://figshare.com/articles/dataset/TF-C_Pretrain_SleepEEG/19930178

5. **Epilepsy**: https://figshare.com/articles/dataset/TF-C_Pretrain_Epilepsy/19930199

6. **FD-A**: https://figshare.com/articles/dataset/TF-C_Pretrain_FD-A/19930205

7. **FD-B**: https://figshare.com/articles/dataset/TF-C_Pretrain_FD-B/19930226

8. **HAR**: https://figshare.com/articles/dataset/TF-C_Pretrain_HAR/19930244

9. **Gesture**: https://figshare.com/articles/dataset/TF-C_Pretrain_Gesture/19930247

10. **ECG**: https://figshare.com/articles/dataset/TF-C_Pretrain_ECG/19930253

11. **EMG**: https://figshare.com/articles/dataset/TF-C_Pretrain_EMG/19930250 



## Date retrieved

June 3rd, 2022

## Contacts

Xiang Zhang(xiang\_zhang@hms.harvard.edu)