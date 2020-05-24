# CDAE python (Tensorflow) Implementation

**Original paper**
 - *Collaborative Denoising Autoencoder (CDAE) (Wu, Y., DuBois, C.,
   Zheng, A. X., & Ester, M. (2016, February)*. Collaborative denoising auto-encoders for top-n recommender systems. In Proceedings of the Ninth ACM International Conference on Web Search and Data Mining (pp. 153-162). ACM.)

**Python (Tensorflow) Implementation**

*Datasets:* politic_old and politic_new (legislative roll-call dataset)

 - **politic_old** : made by Yupeng Gu  (Gu, Y., Sun, Y., Jiang, N., Wang,
   B., & Chen, T. (2014, August). Topic-factorized ideal point
   estimation model for legislative voting network. In Proceedings of
   the 20th ACM SIGKDD international conference on Knowledge discovery
   and data mining (pp. 183-192). ACM.) 1990~2013 legislative rollcall
   dataset (THOMAS)

 - **politic_new** : made by Kyungwoo Song (under the review paper)
   1990~2016 legislative rollcall dataset (from Govtrack.com)

You can download **politic_old** from [https://github.com/gtshs2/NIPEN/tree/master/data/politic_old](https://github.com/gtshs2/NIPEN/tree/master/data/politic_old)
You can download **politic_new** from [https://github.com/gtshs2/NIPEN/tree/master/data/politic_new](https://github.com/gtshs2/NIPEN/tree/master/data/politic_new)

======================================================
## Setup

Check python version (Python 3.7.3)

    python3 --version 
      
Create virtual environment (env)

    python3 -m venv env  
  
Activate virtualenv

	source env/bin/activate

Open source code tensorflow v2.0

    cd src_v2
    
Install dependencies (using requirements file)

	pip3 install -r requirements.txt
    
Run main file

    python3/main.py     
 
 To leave the virtualenv
   
    deactivate