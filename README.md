# GHRS: Graph-based hybrid recommendation system
## Notices
The comments below are from the orginal repository excerpt **the Scripts Added For Reproducing The Result(Including Instruction)**

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ghrs-graph-based-hybrid-recommendation-system/collaborative-filtering-on-movielens-100k)](https://paperswithcode.com/sota/collaborative-filtering-on-movielens-100k?p=ghrs-graph-based-hybrid-recommendation-system)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/ghrs-graph-based-hybrid-recommendation-system/collaborative-filtering-on-movielens-1m)](https://paperswithcode.com/sota/collaborative-filtering-on-movielens-1m?p=ghrs-graph-based-hybrid-recommendation-system)

*"This repo is being updated. Please Watch the repo for upcoming updates and codes"*

Partial implementation for : [GHRS: Graph-based hybrid recommendation system with application to movie recommendation](https://doi.org/10.1016/j.eswa.2022.116850) [pre-print on [arXiv](https://doi.org/10.48550/arXiv.2111.11293)]
## Summary
**GHRS** is a Graph-based hybrid recommendation system for movie recommendation
> Research about recommender systems has emerged over the last decade and comprises valuable services to increase different companies' revenue. Several approaches exist in handling paper recommender systems. While most existing recommender systems rely either on a content-based approach or a collaborative approach, there are hybrid approaches that can improve recommendation accuracy using a combination of both approaches. Even though many algorithms are proposed using such methods, further improvement is still necessary. In this paper, we propose a recommender system method using a graph-based model associated with the similarity of users' ratings in combination with users' demographic and location information. By utilizing the advantages of Autoencoder feature extraction, we extract new features based on all combined attributes. Using the new set of features for clustering users, our proposed approach (GHRS) has gained a significant improvement, which dominates other methods' performance in the cold-start problem. The experimental results on the MovieLens dataset show that the proposed algorithm outperforms many existing recommendation algorithms in terms of recommendation accuracy. [1]


![The framework of the proposed recommendation system. The method encodes the combined features with an autoencoder and creates the model by clustering the users using the encoded features (upper part). At last, a preference-based ranking model is used to retrieve the predicted movie rank for the target user (lower part)](https://raw.githubusercontent.com/hadoov/GHRS/main/Figs/ghrs-structure.png)

The framework of the proposed recommendation system. The method encodes the combined features with an autoencoder and creates the model by clustering the users using the encoded features (upper part). At last, a preference-based ranking model is used to retrieve the predicted movie rank for the target user (lower part)

## Required Libraries
- ScikitLearn
- Tensorflow
- Keras
- Networkx

## Scripts From The Original Repo

[Feature100K.py](https://github.com/hadoov/GHRS/blob/main/Features100K.py): Creates similarity graph between users, extracts graph features and generates the final feature vector by combining the graph features and categorized side information for users (Steps 1, 2 and 3 of GHRS) on dataset MovieLens 100K [2].

[Feature1M.py](https://github.com/hadoov/GHRS/blob/main/Features1M.py): Creates similarity graph between users, extracts graph features and generates the final feature vector by combining the graph features and categorized side information for users (Steps 1, 2 and 3 of GHRS) on dataset MovieLens 1M [2].

## Scripts Added For Reproducing The Result(Including Instruction) 
Run the scripts according to the sequence below to reproduce the result of this paper 

  **1. Feature Extraction**
  This step builds raw graph-based and side information features. This code is original from this repo: https://github.com/hadoov/GHRS/blob/main/Features100K.py\
  
    python Features100K.py
  
  Output: 
  
    x_train_alpha(<alpha>).pkl files in data100k/ (e.g., x_train_alpha(0.01).pkl)

  **2. Dimensionality Reduction with Autoencoder**
  This encodes high-dimensional features into low-dimensional vectors using an autoencoder.
  
    python autoencoder.py
  
  Input: x_train_alpha(0.01).pkl (modify the file path inside autoencoder.py if needed)
  
  Output: 
    
    encoded_features.pkl in data100k/

  **3. User Clustering (KMeans)**
  This clusters the encoded features to group similar users.
  
    python cluster_user.py
  
  Output:
  
    kmeans_diagnostics.png
    
    user_clusters.pkl
  
  **4. Preference-Based Ranking (Rating Estimation)**
  This predicts ratings based on cluster averages and item similarity.
  
    python rank_model.py
  
  Output: 
  
    predicted_ratings.pkl

  **5. Evaluation (RMSE Calculation)**
  This evaluates the predicted ratings against ground truth test set.
  
    python evaluate.py
  
  Output: 
    
    RMSE on test set: ... in console

## Data

     |-datasets
     | |-ml-100k	# MovieLens 100K dataset files
     | |-ml-1m		# MovieLens 1M dataset files
     |-data100k		# Combined features (graph features and side information) for a specific value of alpha for dataset MovieLens 100K
     | |-x_train_alpha(0.005).pkl
     | |-x_train_alpha(0.01).pkl
     | |...
     |-data1m		# Combined features (graph features and side information) for a specific value of alpha for dataset MovieLens 1M
     | |-x_train_alpha(0.005).pkl
     | |-x_train_alpha(0.01).pkl
     | |...
     | ...

## Citation
If you find this research interesting or this repo useful, please cite the main article:

    @article{darban2022ghrs,
      title={GHRS: Graph-based hybrid recommendation system with application to movie recommendation},
      author={Darban, Zahra Zamanzadeh and Valipour, Mohammad Hadi},
      journal={Expert Systems with Applications},
      pages={116850},
      year={2022},
      publisher={Elsevier}
    }

## References
[1] Darban, Z. Z., & Valipour, M. H. (2022). [GHRS: Graph-based hybrid recommendation system with application to movie recommendation](https://www.sciencedirect.com/science/article/abs/pii/S0957417422003025). _Expert Systems with Applications_, 116850.

[2] Harper, F. M., & Konstan, J. A. (2015). [The movielens datasets: History and context](https://dl.acm.org/doi/10.1145/2827872). _Acm transactions on interactive intelligent systems (tiis)_, _5_(4), 1-19.
