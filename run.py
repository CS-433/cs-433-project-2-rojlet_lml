import argparse
import os.path
from models.best_model_implementation import *
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--file', type=str,
                        help='Relative ath to test file (.txt)',
                        required=True)
                     
    parser.add_argument('-o', '--out', type=str,
                        help='Relative path to submission file (.csv)',
                        required=True)


    args = parser.parse_args()

    file_in = args.file # relative dir/file path
    file_out = args.out # relative dir/file path

    
    from scripts.helpers_test import *
    from cleaning.data_cleaning import *
    import tensorflow as tf
    
    test = load_cleaned_data(file_in=file_in, stop_words=False)
    typed = ''
    while(typed!='SVM' and typed!='BERT'):
        which_model = input("which model do you want to try ? type SVM or BERT  : ")
        typed=which_model
    print('Thank you')     
    if (which_model == 'SVM'):
        test_df = create_test_dfs(test)
        X_test = test_df['tweets']
        y_pred = run_SVM(X_test)
    elif(which_model == "BERT"):
        y_pred = run_best_model(test)
    else:
        print('Thank you')

    dir_name = os.path.dirname(__file__)
    create_csv_submission(y_pred, dir_name+file_out)

    print("Prediction done succesfully!")