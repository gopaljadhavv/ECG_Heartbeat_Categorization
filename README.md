
# ECG Heartbeat Categorization

## Overview

#### What is an ECG?
An electrocardiogram (ECG) is a simple test that can be used to check your heart's rhythm and electrical activity.

Sensors attached to the skin are used to detect the electrical signals produced by your heart each time it beats.

These signals are recorded by a machine and are looked at by a doctor to see if they're unusual.

An ECG may be requested by a heart specialist (cardiologist) or any doctor who thinks you might have a problem with your heart, including your GP. That's the result of this test we will analyze.

## Objective
The aims of this project to be able to classify heart disease from heartbeat signal.
## About Dataset
[PTB Diagnostic ECG Database](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)

### Abstract
This dataset is composed of two collections of heartbeat signals derived from two famous datasets in heartbeat classification, the MIT-BIH Arrhythmia Dataset and The PTB Diagnostic ECG Database. The number of samples in both collections is large enough for training a deep neural network.

This dataset has been used in exploring heartbeat classification using deep neural network architectures, and observing some of the capabilities of transfer learning on it. The signals correspond to electrocardiogram (ECG) shapes of heartbeats for the normal case and the cases affected by different arrhythmias and myocardial infarction. These signals are preprocessed and segmented, with each segment corresponding to a heartbeat.

### Content
The PTB Diagnostic ECG Database
- Number of Samples: 14552
- Number of Categories: 2
- Sampling Frequency: 125Hz
Data Source: Physionet's PTB Diagnostic Database

## Implementation
**Libraries:** `Python` `Numpy` `Pandas` `matplotlib` `plotly` `sklearn` 

**Checking Balance of Target variable**
<br>
<img src = "https://github.com/gopaljadhavv/ECG_Heartbeat_Categorization/blob/main/images/newplot.png">
<br>
<img src = "https://github.com/gopaljadhavv/ECG_Heartbeat_Categorization/blob/main/images/newplot_1.png">
<br>
<img src = "https://github.com/gopaljadhavv/ECG_Heartbeat_Categorization/blob/main/images/newplot_2.png">
### Random Forest Classifier

```python
#Random forest classifier
rforest = RandomForestClassifier(random_state=123)
rforest.fit(X_train, y_train)

train_acc = accuracy_score(y_train, rforest.predict(X_train))
test_acc = accuracy_score(y_test, rforest.predict(X_test))

print("Random Forest Classifier \n Train Acc : {} \n Test Acc : {}\n".format(train_acc,test_acc))
print("Confusion Matric Test: \n", confusion_matrix(y_test, rforest.predict(X_test)))
print("Classification Report Test : \n",classification_report(y_test, rforest.predict(X_test)))
print("Overall Precision:",precision_score(y_test, rforest.predict(X_test)))
print("Overall Recall:",recall_score(y_test, rforest.predict(X_test)))
```
```
Random Forest Classifier 
 Train Acc : 1.0 
 Test Acc : 0.9757214841960604

Confusion Matric Test: 
 [[2134   36]
 [  70 2126]]
Classification Report Test : 
               precision    recall  f1-score   support

           0       0.97      0.98      0.98      2170
           1       0.98      0.97      0.98      2196

    accuracy                           0.98      4366
   macro avg       0.98      0.98      0.98      4366
weighted avg       0.98      0.98      0.98      4366

Overall Precision: 0.9833487511563367
Overall Recall: 0.9681238615664846
```
<br>
<img src = "https://github.com/gopaljadhavv/ECG_Heartbeat_Categorization/blob/main/images/ROC.png">


### Lessons Learned
`EDA`
`ML Classification`
`Deep Learning`
`Classification Metrics`

### Feedback

If you have any feedback, please reach out at goplaljadhav061.gj@gmail.com


### 🚀 About Me
#### Hi, I'm Gopal! 👋
I am an Data science Enthusiast and ML practitioner

[1]: https://github.com/gopaljadhavv/gopaljadhavv
[2]: https://www.linkedin.com/in/gopaljadhav/

[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
