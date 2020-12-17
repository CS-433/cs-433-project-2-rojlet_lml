import argparse
import os.path
from models.best_model_implementation import *

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
    
    test = load_cleaned_data(file_in=file_in, stop_words=False)

    test_df = create_test_dfs(test)

    X_test = test_df['tweets']
    print(X_test[:3])

    y_pred = run_SVM(X_test)

    dir_name = os.path.dirname(__file__)
    create_csv_submission(y_pred, dir_name+file_out)

    print("Prediction done succesfully!")