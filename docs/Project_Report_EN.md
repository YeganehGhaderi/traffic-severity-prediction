# Predicting the Severity of Traffic Incidents from Recorded Data

**Course:** Data Mining, University of Tehran
**Instructor:** Dr. Nadi
**Team (Group 32):** Yeganeh Ghaderi, Arvin Bahrami

*English translation of the final project report (originally written in Persian).*

---

## Abstract

Predicting the severity of traffic incidents is difficult because of heterogeneous data, a high rate of
missing values, and severe class imbalance.

In this project, the severity of traffic incidents is treated as a 4-class classification problem on a dataset of
100,000 samples and 34 initial features. In preprocessing, we built a pipeline that extracts features from text
(using Regex and Target Encoding), handles high missing rates (up to 44%), and controls outliers with IQR capping.

In feature selection, we combined three methods (ANOVA, Mutual Information and Extra Trees) by averaging their
ranks, reducing the data from 38 engineered features to exactly 25 optimal features. To deal with imbalance
(minority classes below 1% of the data), we used SMOTE and threshold tuning.

Three classifiers (Decision Tree, Logistic Regression and XGBoost) were implemented and evaluated with a
multi-criteria approach (Macro F1, interpretability and complexity). XGBoost reached a **Macro F1 of 0.58 on the
test set**, partly overcoming the challenge of detecting minority classes, and was selected as the final model.

---

## 1. Introduction

Traffic incidents and the delays they cause are a major problem in modern cities, leading to wasted time,
more pollution and risk to human lives. The main goal of this project is to turn a large amount of crowdsourced
reports into knowledge that can be used to automatically predict the severity of traffic disruptions.

Reaching this goal involves fundamental data-science challenges: combining heterogeneous data (text, numbers and
coordinates); missing values of up to 44% in important columns (where dropping rows would mean losing a large
part of the information); and, most importantly, severe class imbalance. In this dataset, normal events make up
about 80% of the samples, while critical events account for less than 1%. This makes traditional metrics such as
accuracy completely misleading.

To address these challenges we designed a complete pipeline. Without removing any samples, we implemented
imputation and advanced feature engineering (such as extracting keywords from the report text with Regex). Then,
using feature selection, we reduced the data from 38 to 25 optimal features, and finally trained and compared
three different algorithms (from linear models to tree ensembles). The rest of this report describes the technical
details of missing-value handling, the strategies used against class imbalance, and the comparison of the models.

---

## 2. Related Work

Predicting traffic incident severity with machine learning has received a lot of attention in the intelligent
transportation systems (ITS) community. Earlier studies show that the problem is more complex than simple models
can handle.

One study on road-accident data showed that ensemble models such as Random Forest identify severe incidents
better than linear models such as Logistic Regression [1], because they can capture non-linear relationships
between variables.

One of the biggest shared challenges is class imbalance. Related work showed that when fatal accidents are only a
tiny fraction of the data, accuracy is not a valid metric, and recommended oversampling techniques such as SMOTE
or class re-weighting [2] so that the model is forced to learn minority patterns. In addition, recent work uses
text data (such as police reports or traffic tweets) as a very important predictive feature, and showed that
word-level feature extraction and named-entity recognition (for street names and accident types) can improve
prediction accuracy considerably [3].

What distinguishes this project from earlier work is its integrated approach: combining text feature extraction,
simultaneous handling of missing values and outliers, and several balancing strategies (SMOTE with different
ratios) across three models, together with a careful comparison of interpretability, computational cost and
minority-class prediction quality.

---

## 3. Methodology

### 3.1 Data splitting for training and evaluation

To prevent data leakage and ensure an unbiased evaluation, we split the original dataset (100,000 samples) into
three parts **before any preprocessing**. Because of the severe class imbalance, we used stratified splitting so
that class proportions stay the same in all sets. The split was 70% for training (70,024 samples, after removing
duplicates), 15% for validation (14,960 samples) and 15% for the final test (15,000 samples). The validation set
was used to tune hyperparameters, and the test set only to report the final result.

### 3.2 Exploratory data analysis (EDA)

The dataset has 100,000 samples (rows) and 34 initial features (columns). Each sample is a traffic-disruption
report containing geographic coordinates and time of occurrence, weather conditions and a text description. The
target variable is column `y`, which divides severity into 4 classes (from normal to critical). The problem is
therefore a **multi-class classification** problem. The main findings of this stage:

- **Target distribution and class imbalance:** class 2 (normal traffic) made up about 80% of the data, while
  class 1 (critical) made up less than 1%.
- **Many missing values:** feature `f6` had 44% missing values, and `f20` and `f15` had about 25-28%.
- **Constant columns:** `f12` (country, containing only "US") and `f31` (a boolean that is always False) carried
  no information.
- **Distributions, outliers and correlation:** histograms of variables such as "incident duration" revealed
  extreme outliers (e.g. an incident lasting more than 1 million minutes), which made IQR capping necessary. The
  correlation matrix also showed no significant linear correlation between the numeric features, which pushed us
  toward non-linear models.

### 3.3 Cleaning and preprocessing

We defined a custom pipeline with deduplication that applies the following steps in order:

1. **Duplicate removal:** 16 exactly duplicated rows were removed from the training set.
2. **Regex-based text feature extraction:** instead of heavy methods such as TF-IDF (which would create thousands
   of sparse columns and cause overfitting), we used Regex. The text descriptions in `f8` and `f9` were analyzed to
   extract key concepts (for example: is the route a highway? does the text contain words like "accident" or
   "blocked"? is slow traffic reported?). This created 5 very efficient boolean features.
3. **Time features:** from the date/time columns (`f2`, `f3`) we derived new features such as incident duration in
   minutes, start hour, month and day of week.
4. **Missing values:** depending on column distribution, `SimpleImputer` was used with the median strategy for
   columns with outliers and the mean strategy for normally distributed columns. For categorical text columns,
   missing values were replaced by the overall mean of the target during Target Encoding.
5. **Outliers:** `IQRCapper` with a factor of 1.5 was used to cap outliers in columns such as temperature and
   humidity, to prevent distortion of the model.
6. **Scaling and encoding:** `RobustScaler` for numeric data with outliers, `StandardScaler` for normal data and
   `MinMaxScaler` for time. For categorical variables with many categories (such as city and state), Target
   Encoding was used to avoid exploding the dimensionality.

The output of this stage turned the 34 initial features into 38 engineered features, fully numeric and with no
missing values.

### 3.4 Feature selection

The project requires at most 25 features for the final modeling. We used a "combined ranking" system. First, columns
with constant values (`f31`, `f12`) and column `f6` (more than 40% missing) were removed. Then, on the remaining 35
features, we tried three methods:

- **Mutual Information:** measures the non-linear dependency between each feature and the target.
- **ANOVA F-score:** measures the linear dependency between features and the target.
- **Extra Trees feature importance:** a strong tree model that computes each column's importance from impurity
  reduction (Gini impurity).

Each feature received a rank from each of the 3 methods, and we averaged these ranks. Features with the worst
average rank (such as `f16`, `f17`, `f20`) were dropped, and the top 25 features with the highest predictive
power were kept.

### 3.5 The three models

Our principle for choosing the 3 models was "simple model -> complex model -> different model":

**1) Decision Tree** was chosen as a simple non-linear model with high interpretability and fast training. It
builds "if-then" rules and recursively splits the feature space by impurity reduction (e.g. Gini). Its biggest
advantage is interpretability; however, it tends strongly to overfit, and its tree structure may not generalize
well on the synthetic (SMOTE-generated) data.

**2) XGBoost (Extreme Gradient Boosting)** was chosen as a complex tree model. It uses ensemble learning with
post-pruning to prevent overfitting and gradient-based optimization, and is one of the strongest algorithms for
tabular data. This boosting algorithm builds hundreds of weak trees sequentially, each new tree correcting the
residual errors (gradients) of the previous one. XGBoost approximates the loss with a second-order Taylor
expansion and adds a regularization term to the objective that limits the complexity of the trees (depth and
number of leaves). The three main reasons for choosing it were:

- *Robustness to imbalance:* it naturally focuses on samples the previous model got wrong (the critical classes 1 and 4).
- *Understanding interactions:* it can learn the joint effect of features (e.g. rain + being on a highway).
- *Overfitting control:* through its regularization parameters.

Its weaknesses are lower interpretability and longer training time.

**3) Logistic Regression** was chosen as a linear and different model that assumes linear decision boundaries
between classes. It was chosen as a **baseline**, computing class probabilities with the Softmax function.
Although its training speed and interpretability (through coefficients) are high, the linearity assumption was a
major obstacle on this dataset. To compensate and to deal with imbalance, we had to add an inexpensive model-external
technique: dynamic decision **threshold tuning**.

### 3.6 Training and tuning

To overcome the bias of the models toward the majority class (class 2), the following strategy was implemented:

- **SMOTE configurations:** instead of one simple setting, a dictionary of 4 different configurations (from no
  change to partially balancing the rare classes) was defined to find the best synthetic-data generation mode for
  each model.
- **Dynamic threshold tuning (Logistic Regression only):** to compensate for the weakness of this linear model,
  instead of a fixed 0.5 threshold, the raw output probabilities were multiplied by different coefficients (e.g.
  a factor of 10 for class 4) so that the model pays attention to minority classes.
- **Hyperparameter tuning:** a dedicated search space was defined for each model (regularization parameter `C` for
  logistic regression, tree depth for Decision Tree, and learning rate / subsampling for XGBoost).

All of these were combined in a nested way with the SMOTE strategies, and the winning combination was chosen by
the highest **Macro F1** on the validation data.

### 3.7 Comparison and the final model

The final model was not chosen based on a single number:

- **Logistic Regression:** despite its speed and interpretability, because of the linear limitation it could not
  learn complex boundaries even with threshold changes (Macro F1 ≈ 0.48).
- **Decision Tree:** because it is prone to overfitting, it could not make good use of the SMOTE data
  (Macro F1 ≈ 0.49).
- **XGBoost:** although examining 972 combinations (4 SMOTE strategies × 243 hyperparameter sets) was
  time-consuming, its internal mechanisms prevented overfitting.

XGBoost was finally selected. The main reason was the clearly significant gap (about 9-10 absolute points) in
Macro F1. In critical problems such as traffic-incident prediction, correctly detecting minority classes matters
far more than saving processing time, so this complexity is fully justified.

---

## 4. Analysis and Results

Because of the severe class imbalance, **accuracy** is misleading (always predicting the majority class gives a
fake 80% accuracy). The evaluation therefore used the following metrics:

- **Precision, Recall and F1-score:** to measure the balance between "correctness of positive predictions" and
  "ability to find the real samples" (especially for critical classes).
- **Macro average (our main metric):** the simple average of F1 over all classes. Because it gives equal weight to
  majority and minority classes, it is a realistic indicator of how well the model detects critical events.
- **Weighted average:** weighted by class size, so it is dominated by the large class 2; it is complementary.

### 4.1 Comparison of the three models (test set)

| Model | Accuracy | Macro F1 | Weighted F1 | Inference time (s) |
|---|---|---|---|---|
| Logistic Regression (tuned threshold) | 0.7721 | 0.4783 | 0.8006 | 0.0214 |
| Decision Tree (SMOTE + tuned) | 0.8082 | 0.4910 | 0.8161 | 0.0140 |
| **XGBoost (SMOTE + tuned)** | **0.8437** | **0.5772** | **0.8509** | 0.2063 |

### 4.2 Effect of feature selection

To verify that dimensionality reduction was successful, we measured the performance of a baseline model
(Random Forest) before and after removing features:

| Feature set | Number of features | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|---|
| All candidate features | 35 | 0.8209 | 0.5213 | 0.8317 |
| Selected features | 25 | 0.8237 | 0.5266 | 0.8343 |

### 4.3 In-depth analysis of the final model (XGBoost)

Classification report on the test data:

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 1 | 0.33 | 0.44 | 0.38 | 127 |
| 2 | 0.94 | 0.87 | 0.90 | 11,947 |
| 3 | 0.62 | 0.83 | 0.71 | 2,519 |
| 4 | 0.30 | 0.33 | 0.31 | 407 |
| Accuracy | | | 0.84 | 15,000 |
| Macro avg | 0.55 | 0.62 | 0.58 | 15,000 |
| Weighted avg | 0.87 | 0.84 | 0.85 | 15,000 |

![Confusion matrix of the final model on the test set](../outputs/final_model_confusion_matrix.png)

**Strengths**

- **Class 2:** the model performs very well (F1 = 0.90). In the confusion matrix, of the 11,947 real samples of this
  class, more than 10,000 were detected correctly.
- **Class 3:** with F1 = 0.71 the performance is acceptable. The model detects incidents caused by moderate
  rainfall or small incidents on local streets well.

**Weaknesses**

- **Class 1:** F1 = 0.38 is the model's weakest point. The confusion matrix shows the model could not find a clear
  boundary between severe incidents and ordinary delays, and a large part of this critical class was wrongly
  predicted as class 2 or 3. The main reason is the severe lack of training data (less than 1%) and the
  similarity of the environmental features of these incidents to other incidents.
- **Class 4:** a similar situation to class 1 (F1 ≈ 0.31). Sometimes the difference between a complete blockage
  and a long-lasting incident cannot be separated from physical data (such as temperature or wind speed), and only
  the report text can tell them apart; the model is limited in deep understanding of natural-language semantics.

**Two example test cases**

*Successful case (correct prediction: class 3).* Feature `f8_is_highway` = 1 (occurred on a highway),
`f8_is_slow_traffic` = 1 (slow traffic reported in the text), and `f21` (weather) = "Light Rain". The model
correctly learned that "light rain + highway" usually leads to slowdowns and moderate delay (class 3) and rarely
causes a full crisis.

*Failed case (true class 1, model predicted class 2).* Feature `f8_is_severe_crash` = 0 (the word "crash" was not
in the report text), weather was favorable, but the geographic coordinates indicate a traffic blind spot. The reason
for the error is clear: because the report text did not use critical keywords and the weather was normal, the only
way to detect the severity was a deep understanding of the location, and our current features (such as `f10`, the
city name) cannot extract this level of complexity. This shows the need for more advanced spatial features (such as
linking to traffic heatmaps) in future work.

---

## 5. Conclusion

In this project, we presented a comprehensive framework for handling real data-science challenges, including
heterogeneous data, high missing rates (up to 44%) and severe class imbalance. Our most important
achievements were showing that accuracy is misleading, and successfully reducing the data from 34 raw features to
25 optimal features without a drop in performance.

Comparing the algorithms showed that linear models (even with sophisticated tricks such as threshold tuning) do
not have the capacity to learn the complex decision boundaries of this dataset. Finally, XGBoost, with the optimal
combination of SMOTE and careful tuning, won with a **Macro F1 of 0.58**. The final takeaway is that in applied
problems the model choice should not rest on a single number: we chose XGBoost because, despite its higher
complexity, it achieved an unmatched balance between detecting majority and minority classes, which is vital and
justified for managing traffic crises.

### Future work

- **Advanced NLP:** replacing Regex features with word embeddings could capture the semantics of the report text
  better and greatly improve detection of critical classes.
- **Spatial data and road topology:** using graph neural networks (GNNs) instead of independent coordinates, to
  model the real structure of the road network and the effect of topology on traffic severity.
- **Operational interpretability:** implementing SHAP values so that users (such as police or city management) can
  be shown transparently which features led to an incident being flagged as critical.
- **Multi-label formulation:** turning the problem from multi-class to multi-label could better reflect the
  combined nature of real incidents (e.g. "complete blockage" and "severe damage" at the same time).

---

## 6. Team Contributions

- **Arvin Bahrami:** data exploration and familiarization, EDA, feature selection, and implementation of the
  Logistic Regression model.
- **Yeganeh Ghaderi:** preprocessing, implementation of the Decision Tree and XGBoost models, final evaluation,
  and writing the final report.

Although tasks were divided this way, at the end of each stage we debugged the code together and made changes where
needed.

---

## References

The Persian report cites three prior studies ([1] ensemble models vs. linear models for severity prediction,
[2] class imbalance and oversampling in traffic data, [3] text and named-entity features for severity prediction),
but the reference list itself is not included in the report file. Add the full references here before publishing.
