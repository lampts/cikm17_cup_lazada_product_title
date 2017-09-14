import os, sys, math
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr

'''
ensemble result
'''
def merge_result(home_folder, input_folders, weights, target_name, verbose = 0):
    df_inputs = []
    for i in range(len(input_folders)):
        filename = os.path.join(home_folder, input_folders[i], target_name + "_test.predict")
        df_input = pd.read_csv(filename, header=None)
        df_input.columns = ["preds_" + str(i)]
        df_inputs.append(df_input)

    df_result = pd.concat(df_inputs, axis=1)
    df_result["preds"] = df_result["preds_0"] * weights[0]
    for i in range(1, len(weights)):
        df_result["preds"] = df_result["preds"] + df_result["preds_" + str(i)] * weights[i]

    df_result["preds"] = df_result["preds"] / sum(weights)

    print(df_result[:20])
    output_filename = os.path.join(home_folder, "output", target_name + "_test.predict")
    df_result.to_csv(output_filename, columns=["preds"], index=False, header=None)

'''
calibrate result based on mean of training data
'''
def calibrate_results(home_folder, input_folder, is_concise=True, using_train_mean=False):

    if is_concise:
        label_filename = os.path.join(home_folder, "input", "conciseness_train.labels")
        output_filename = os.path.join(home_folder, input_folder, "conciseness_test.predict")
    else:
        label_filename = os.path.join(home_folder, "input", "clarity_train.labels")
        output_filename = os.path.join(home_folder, input_folder, "clarity_test.predict")

    if using_train_mean:
        df_label = pd.read_csv(label_filename, header=None)
        df_label.columns = ["label"]
        label_mean = df_label["label"].mean()
    else:
        if is_concise:
            label_mean = 0.658592393877
        else:
            label_mean = 0.923622939504

    df_output = pd.read_csv(output_filename, header=None)
    df_output.columns = ["pred"]
    output_mean = df_output["pred"].mean()

    adjusted_value = label_mean - output_mean
    print("Adjusted value {}".format(adjusted_value))

    df_output["result"] = df_output["pred"] + adjusted_value
    df_output.loc[(df_output["result"] > 1.0), "result"] = 1.0
    df_output.loc[(df_output["result"] < 0.0), "result"] = 0.0
    print(df_output[:10])

    # write to file
    if is_concise:
        output_filename = os.path.join(home_folder, "output", "conciseness_test.predict")
    else:
        if using_train_mean:
            # write back to the input file
            output_filename = os.path.join(home_folder, input_folder, "clarity_test.predict")
        else:
            output_filename = os.path.join(home_folder, "output", "clarity_test.predict")
    df_output.to_csv(output_filename, columns=["result"], index=False, header=None)

#===============================================================================
HOME_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(HOME_DIR, "output")

conciseness_predictions = [
    "conciseness21_3298/output", "conciseness21_3301/output",
    "conciseness22_3298/output", "conciseness22_3301/output"
]
conciseness_weights = [
    1.0, 1.0,
    1.0, 1.0,
]
'''
merge_result(HOME_DIR, ["hieuvq/hieuvq-1-input", "hieuvq/run08_ensemble", "saigon/run10"], weights = [2.0, 1.0, 1.5], target_name="conciseness")
calibrate_results(HOME_DIR, "output", is_concise=True)

merge_result(HOME_DIR, ["hieuvq/hieuvq-6-input", "saigon/run08"], weights = [1.1, 1.0], target_name="clarity")
calibrate_results(HOME_DIR, "output", is_concise=False)
'''

#deflate
merge_result(HOME_DIR, ["hieuvq/run01_3298", "hieuvq/run01_3301", "saigon/run01", "hieuvq/run08_3298", "hieuvq/run08_3301", "saigon/run10"], weights = [5.0, 5.0, 12.0, 4.2, 4.2, 12.6], target_name="conciseness")
calibrate_results(HOME_DIR, "output", is_concise=True)

calibrate_results(HOME_DIR, "hieuvq/run02_3244", is_concise=False, using_train_mean=True)
merge_result(HOME_DIR, ["hieuvq/run02_3244", "saigon/run06", "saigon/run08"], weights = [1.1, 3.3, 4.0], target_name="clarity")
calibrate_results(HOME_DIR, "output", is_concise=False)

#calibrate_results(HOME_DIR, "hieuvq/hieuvq-427-input", is_concise=True)

#merge_result(HOME_DIR, ["hieuvq/run07_ensemble", "saigon/run07"], weights = [1.0, 1.5], target_name="conciseness")
#merge_result(HOME_DIR, ["hieuvq/run02_ensemble", "saigon/run06"], weights = [1.0, 3.0], target_name="clarity")
