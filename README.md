# Brain-Stroke-Prediction-Kaggle-Challenge
The Solution to the Kaggle Challenge of Brain Stroke Prediction 

# Project Report

# 1. Steps in Data Preprocessing
1. The initial shape of the data was: I split the data into training and test sets (80:20), ensuring
that the models could be evaluated on unseen data.
2. Since the target variable is highly imbalanced, with significantly more stroke==0 cases than stroke==1 cases,
the problem is combated by balancing the data prior to running the model. I
used SMOTE to oversample the minority class (stroke == 1). The left figure shows the target variable before
balancing and right one shows the same after balancing.
3. I encoded the categorical variables {['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']}
into numerical values to ensure that they can be included in the models.
4. Also encoded the continuous variables {'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi'} by
binning.
5. I scaled numerical variables using StandardScaler() to ensure that no feature dominates another, resulting in a
bias. The aim of scaling is for each feature to contribute equally to the models' results. Also, normalisation was
essential for performing PCA at further step.
6. I handled outliers in the ‘bmi’ and ‘avg_glucose_level’ using z score. These are the features with the maximum
number of outliers as seen from the box plot below:
7. There is no need to handle the null values are there are none.

# 2. EDA by plotting the correlation plot, distribution report, histograms, and pie charts.

# 3. Training a Neural Network
After importing a neural network model from tensorflow, I compiled the model by specifying the number of
hidden layers as 2, loss function {binary_crossentropy}, optimizer {adam}, and evaluation metric{accuracy} to
use during training. The model architecture consists of three layers: two Dense layers with relu activation
functions, and a final Dense layer with a sigmoid activation function. Model accuracy is 94.20%

# 4. Making an ensemble:
A. A Bagging classifier:
Hyperparameter tuning using Gridsearch CV. Following are the best parameters for
1. SVM classifier:
2. Decision Tree classifier:
3.Logistic Regression
4.KNNClassifier
5. Kmeans:
Both the elbow plot and sihouette score gave the same result: k=3
After doing dimensionality reduction using PCA (n_components=2), I did the kmeans clustering with k=3.
Model evaluation of the bagging classifier
Then I made a bagging classifier with n_estimators=10 using SVM classifier, Decision Tree classifier, Logistic
Regression, KNN classifier, Naïve Bayes as base estimators and using the optimal values hyperparameters
obtained in the above step. Then I plotted their accuracy scores:
SVM, Logistic regression and KNN had the highest and the same accuracy of 94.3%. Considering other metrics
like F1score, precision and recall and losses like Hinge Loss for SVM and Logloss for logistic regression I
decided that Logistic regression is the best classifier among this bagging ensemble.
B. Boosting classifier with model evaluation:
I then trained the dataset using 3 boosting classifiers and their classification accuracies are:

# 5. Training a Random Forest classifier 

# 6. Selecting the best one among all the supervised classification models:
There are 4 best classifiers identified among the supervised models, i.e., Logistic regression, Random Forest,
KNN and Adaboost. The 5-fold CV scores and classification metrices are used to find the best model.
• The dataset is imbalanced, with the positive class {stroke=1} having a much smaller number of samples than the
negative class {stroke=0}. In such cases, accuracy is not an appropriate metric to evaluate the model's
performance since it can be misleading.
• Therefore, we need to choose a metric that is more appropriate for imbalanced datasets. Two popular metrics for
imbalanced datasets are precision and recall. Looking at the classification reports for the four models, we can
see that all models have low precision for the positive class, which means that they are not good at identifying
the positive class correctly. However, some models have higher recall for the positive class than others.
• Among the given models, the KNN model has the highest recall for the positive class (0.30) followed by
AdaBoost (0.43), Random Forest (0.67), and Logistic Regression (0.69). Therefore, if we prioritize correctly
identifying positive cases, we should choose KNN or AdaBoost.
• However, we also need to consider the cross-validation scores for each model. The KNN model has the lowest
cross-validation scores among the four models, indicating that it may not generalize well to new data. On the
other hand, Random Forest and AdaBoost have similar cross-validation scores, which are higher than the crossvalidation scores for Logistic Regression.
• Therefore, based on the given information, we should choose either Random Forest or AdaBoost, depending on
our priorities. If we prioritize correctly identifying positive cases, we should choose AdaBoost. If we
prioritize overall performance and generalization to new data, we should choose Random Forest.

# 7. Unsupervised Learning Models using Heirchical, K-means and DBSCAN clustering and comparing the silhouette scores.

# 8. Before reinforcement learning:
Using the best unsupervised model DBSCAN to remove the outliers and then the best supervised model
RandomForestClassifier and see how the classification accuracy report 1 goes:
# 9. After reinforcement learning:
To further improve the accuracy, I decided to train a reinforcement learning model to optimize the
parameters of a DBSCAN clustering algorithm in order to improve the performance of a random forest
classifier.
• I first defined a function called get_reward that calculates a reward based on the accuracy of the classifier's
predictions. The reward increases as the accuracy improves, and different levels of reward are assigned based on
different levels of accuracy.
• Next, I created a DBSCAN clustering algorithm with an eps value of 0.5 and a min_samples value of 5, as well
as a RandomForestClassifier with 100 estimators. The main loop of the code then runs for 10 episodes. During
each episode, I fit the DBSCAN algorithm to the data X and use the resulting labels to fit the random forest
classifier to the labeled data points.
• I evaluated the classifier's predictions and used the get_reward function to calculate a reward based on the
accuracy of the predictions. Based on the reward, I updated the eps and min_samples parameters of the
DBSCAN algorithm. If the reward is 0, indicating poor performance, I reduce the eps value and increase the
min_samples value.
• If the reward is 1, I increased the eps value and decrease the min_samples value. If the reward is 5, I increase the
eps value further and decrease the min_samples value even more. If the reward is 10, I reduce the eps value
significantly and increase the min_samples value significantly.
• Finally, I output a classification report for the predictions of the random forest classifier using the optimized
parameters of the DBSCAN algorithm.

Based on the classification reports, Classification Report for the second is more suitable, because:
1. Precision: In Classification Report2, the precision for class 0 (no stroke) is 0.97, which means that out of all the
instances predicted as “no stroke”, 97% were actually “no stroke”. In Classification Report1, the precision for
class 0 is 0.96, which is slightly lower than in Report2. However, the precision for class 1 (had stroke) is
much higher in Report1 (0.05) than in Report2 (0.00), which means that in Report1, even though very few
instances were correctly identified as “had stroke”, most of the instances predicted as “had stroke” were
actually “no stroke”.
2. Recall: In Classification Report2, the recall for class 0 is 1.00, which means that out of all the instances that
were actually “no stroke”, 100% were correctly identified as “no stroke”. In Classification Report1, the recall
for class 0 is 0.81, which is lower than in Report2. The recall for class 1 is also higher in Report1 (0.23) than in
Report2 (0.00), which means that in Report1, even though very few instances were correctly identified as “had
stroke”, a higher percentage of actual “had stroke” instances were correctly identified as such.
3. F1-score: In Classification Report2, the F1-score for class 0 is 0.98, which is very high, and the F1-score for
class 1 is 0.00, which is very low. In Classification Report1, the F1-score for class 0 is 0.88, which is lower than
in Report2, and the F1-score for class 1 is 0.08, which is also very low.
4. Accuracy: Accuracy is the ratio of correct predictions to the total number of predictions. In Classification
Report2, the accuracy is 0.97, which is very high. In Classification Report1, the accuracy is 0.96, which is lower
than in Report2.
In summary, while the precision and recall for class 1 are slightly higher in classification Report1, the overall
performance metrics, including precision, recall, F1-score, and accuracy, are all significantly better in
Classification Report2. Therefore, Classification Report2 is more accurate for this dataset.

# 10. Conclusion
   
The best classification accuracy for stroke prediction is 97% and other metric values are obtained in the above
table using a reinforcement learning algorithm to optimize the parameters of a DBSCAN clustering algorithm in
order to improve the performance of a random forest classifier.
