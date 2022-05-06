# Expedia Group ECML/PKDD 2022 challenge

## Zero-shot Cross-brand Lodging Recommendation

In online platforms it is often the case to have multiple brands under the same group which may target different 
customer profiles. For example, in the hospitality domain Expedia Group has multiple brands like Brand Expedia, 
Hotels.com, Orbitz or Vrbo. In this context, being able to provide cross-brand recommendations to travelers is an 
important task as it can improve traveler experience across different point of sales. In this challenge we propose a 
cross-brand recommendation task where the participants will be provided with traveler actions in a source brand 
(e.g. property clicks) and asked to predict actions in target brands. The objective is to improve the recommendations 
in target brands using the data from a source brand.

## Organizers

- Adam Woznica, Expedia Group
- Ioannis Partalas, Expedia Group
- Jan Krasnodebski, Expedia Group

## Important Dates

- May 5, 2022: Training and holdout data released
- June 20, 2022: Test data released
- June 22, 2022: Participants submit the predictions
- July 17, 2022: Paper submission
- July 30, 2022: Author notification
- September XX, 2022, 2022: Challenge session at ECML/PKDD

# Legal

- This *code* is available under the [Apache 2.0 License](http://www.apache.org/licenses/LICENSE-2.0.html).
- This *data* is available under the [Creative Commons Attribution-NonCommercial 4.0 International License 
(CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/). Detailed terms of use for Expedia Group datasets can 
 be found in [TERMS_OF_USE_DATA.md](TERMS_OF_USE_DATA.md).

# Description of the Data

The Expedia Group dataset consists of a random sample of global lodging clicks from consumers in multiple countries 
across multiple brands and tens of thousands of destinations. A click is defined as a visit on a property description 
page that can originate from lodging sort, search engine (e.g. Google) or from lodging meta services (e.g. Trivago).
A property refers to one of over a million hotels, vacation rentals, apartments, B&Bs, hostels and other properties 
appearing on Expedia Group brands. We will also provide auxiliary property description such as star rating, 
user review rating, popularity index, amenity indicators etc.  

## Main Data

The main data consists of a sequences of property ids per user, ordered by event timestamp. Each sequence is assigned 
a unique id. 
```
1 2230829,5612184
5 2001935,3644546,6869346,5848273
7 1707585,517567,6931241,6197397
...
```

Training data, `train.tsv.gz`, is limited to Brand A customers and span 12,709,329 unique users. Holdout data correspond to four 
brands B, C, D and E brands and span 12,191,159 unique users. The holdout data is split into a smaller part, `holdout.tsv.gz`, that will 
be released together with the training data (133,836 users) and a larger part, `test.tsv.gz`, that will be released toward the end 
of the competition (12,057,323 users). Train and holdout data contain 1,304,763 unique properties. 

Sequences of clicked properties were deduplicated by eliminating duplicate copies of repeating property ids. The data 
span a one-month period and contain a random sample of consumers who made at least two clicks (after deduplication). 
Consumers who clicked more than 200 properties during this period (after deduplication) are excluded. Data is limited 
to users with valid property ids of all the clicked properties. 

## Property Attributes

In addition to the main dataset we also released a property attributes dataset, `properties.tsv.gz`. 

Property attributes would likely boost predictive performance of proposed approaches. These attributes would allow us
to go beyond a list of identifiers and generalize over *similar* properties.

| Attribute Name | DataType       | Description                                                                                  |
|----------------|----------------|----------------------------------------------------------------------------------------------|
| prop_id        | Long           | It matches prop_id from sequences.                                                           |
| type           | Integer        | The property type (anonymized); less frequent property types are merged into one indicator. |
| star_rating    | Integer        | The star rating of the hotel, from 1 to 5.  A null indicates the property has no stars, the star rating is not known or cannot be publicized. |
| review_rating  | Integer        | The mean customer review score for the property on a scale out of 5, rounded to nearest integers. A null means there have been no reviews. |
| location_score | Float          | Location scores, the higher the score the better location of a property. |
| amenity_airconditioning | Boolean ||
| amenity_airporttransfer | Boolean ||
| amenity_bar | Boolean ||
| amenity_freeairporttransportation | Boolean||
| amenity_freebreakfast | Boolean||
| amenity_freeparking | Boolean||
| amenity_freewifi | Boolean||
| amenity_gym | Boolean||
| amenity_highspeedinternet | Boolean||
| amenity_hottub | Boolean||
| amenity_laundryfacility | Boolean ||
| amenity_parking | Boolean ||
| amenity_petsallowed | Boolean ||
| amenity_privatepool | Boolean ||
| amenity_spaservices | Boolean ||
| amenity_swimmingpool | Boolean ||
| amenity_washerdryer | Boolean ||
| amenity_wifi | Boolean ||

# Evaluation

The submitted models will be evaluated on *hits@k* metric which is defined as the number of times the next clicked 
item appears at the top k predicted items. We will set k=5.

# Baselines

We release an implementation of a simple baseline approach as well as the evaluation *hits@k* metric. The baseline 
implements a simple *Markov model* that calculates the transition matrix based on a window of size 1 to 
calculate the transition probabilities. During prediction the baseline uses only the last click in the session to 
propose the next clicked hotel. The baseline and evaluation metric are implemented in Python and Tensorflow.

In order to run the baseline you can use the following command:
```sh
python baseline_evaluator.py --train_path path_to_train_file --eval_path path_to_test_file
```

## Software requirements

The code has been tested with `Python 3.8` and requires `Tensorflow>=2.0`, `pandas>=1.4`, `numpy>=1.18` and 
`scipy>=1.4`. We also provide a Python requirements file in order to ease the setup of the environment.

# Data splits for clicks

We provide three files:
- A training dataset `train.tsv.gz` that will be used for training the models. It is the **only** dataset that should be 
  used for training. Recall, that we aim to assess zero-shot learning approaches.
- A holdout dataset `holdout.tsv.gz` that should not be used for model training.
- The final `test.tsv.gz` that we will upload towards the end of the competition and on which the participants will have 
  to predict the next clicked hotel. These predictions will be used to assess the final performance.

## Format of the datasets
### Train dataset

The format of the dataset is tsv and contains tab-separated unique `id` and a sequence of clicks (`clicks`).
```
id  clicks
1   2230829,5612184
5   2001935,3644546,6869346,5848273
7   1707585,517567,6931241,6197397
```

### Holdout dataset

The holdout dataset is also tab-separated and has the `id` column, a sequence of clicks (`clicks_no_last`) and 
the next click to be predicted (`clicks_last`).
```
id      clicks_no_last  clicks_last
58      6580953 228743
120     7301984 5236170
183     6433012,4201850,6611247 5576105
224     2036363,1992045,3196164 6499129
```

### Test dataset

The test dataset is also tab-separated and has the following format:
```
id      clicks_no_last
58      6580953
120     7301984
183     6433012,4201850,6611247
224     2036363,1992045,3196164
```

Note that the next clicked hotel is missing and is the one to be predicted by participants.

# Submission file

The authors should submit a zipped file that contains the predicted next clicked hotel in the sequence. It should be 
tab-separated and have the following format:
```
id      prediction
58      6580953
120     7301984
183     6433012
```

Copyright 2022 Expedia, Inc.
