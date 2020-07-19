
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils.Evaluation.Evaluator import EvaluatorHoldout

from skopt.space import Real, Integer, Categorical
from utils.ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from utils.ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
import os

def hyperparams_tuning(recommender_class, URM_train, URM_validation, URM_test):

    # Step 1: Import the evaluator objects
    
    print("Evaluator objects ... ")
    
    cutoff = 5
    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[cutoff])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[cutoff, cutoff+5])
    # evaluator_validation_earlystopping = EvaluatorHoldout(URM_train, cutoff_list=[cutoff, cutoff+5], exclude_seen=False)
    
    # Step 2: Create BayesianSearch object
    print("BayesianSearch objects ... ")
    
    parameterSearch = SearchBayesianSkopt(recommender_class,
                                          evaluator_validation=evaluator_validation,
                                          evaluator_test=evaluator_test)
    
    # Step 3: Define parameters range   
    print("Parameters range ...") 

    # n_cases = 8 
    # n_random_starts =  int(n_cases / 3) # 5
    n_cases = 2
    metric_to_optimize = "MAP"
    output_file_name_root = "{}_metadata.zip".format(recommender_class.RECOMMENDER_NAME)

    hyperparameters_range_dictionary = {}
    hyperparameters_range_dictionary["topK"] = Integer(5, 1000)
    hyperparameters_range_dictionary["l1_ratio"] = Real(low=1e-5, high=1.0, prior='log-uniform')
    hyperparameters_range_dictionary["alpha"] = Real(low=1e-3, high=1.0, prior='uniform')


    # earlystopping_keywargs = {"validation_every_n": 5,
    #                           "stop_on_validation": True,
    #                           "evaluator_object": evaluator_validation_earlystopping, # or evaluator_validation
    #                           "lower_validations_allowed": 5,
    #                           "validation_metric": metric_to_optimize,
    #                           }

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={}  # earlystopping_keywargs 
    )
    
    output_folder_path = "../results/result_experiments/"
    
    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    # Step 4: run

    best_parameters = parameterSearch.search(recommender_input_args, # the function to minimize
                                                parameter_search_space=hyperparameters_range_dictionary, # the bounds on each dimension of x
                                                n_cases=n_cases,   # the number of evaluations of f
                                                n_random_starts=1, # the number of random initialization points
                                                #n_random_starts = int(n_cases/3),
                                                save_model="no",
                                                output_folder_path=output_folder_path,
                                                output_file_name_root=output_file_name_root,
                                                metric_to_optimize=metric_to_optimize
                                            )
                                
    print("best_parameters", best_parameters)

    # Step 5: return best_parameters
    # from utils.DataIO import DataIO
    # data_loader = DataIO(folder_path=output_folder_path)
    # search_metadata = data_loader.load_data(recommender_class.RECOMMENDER_NAME + "_metadata.zip")
    # print("search_metadata", search_metadata)
    
    # best_parameters = search_metadata["hyperparameters_best"]

    return best_parameters