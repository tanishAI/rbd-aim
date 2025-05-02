import os
import argparse
from rbdaim.train_model import run_experiment, calculate_prediction, run_experiment

def get_model_config():
    config = {"num_classes": 12,
              "model_type": "CDR",
              "epochs": 4,
              "lr":2e-5,
              "device":"cuda:0"
              }
    return config

def train_and_eval_model(dataset, config, output_path):
    for i in range(10):
        data_= run_experiment(dataset,
        	config,
        	cv_id=i,
        	model_type="CDR",
        	ensemble_id=0,
        	full_model=False,
        	out_dir=output_path, 
        	num_classes = config["num_classes"])
    dataset_emb = calculate_embeddings(dataset, config)
    test_classifier_all_cv_models(m_dir = output_path, dataset=dataset_emb)


def parser_args():
	parser = argparse.ArgumentParser(description='Prediction of RBD epitopes')
    parser.add_argument('--input-data', '-i',
    	default="./dataset/datset_production.pkl", help='Path to dataset in pkl format')
    parser.add_argument('--output-path', '-o', 
    	default="./predictions", help='Path to ouput directory to save dataset with embeddings and predictions')
    return parser.parse_args()


def main():
	args = parser_args().
	dataset = pickle.load(open(args.input_data, 'rb'))
	config = get_model_config()
	train_and_eval_model(dataset, config, args.output_path)


if __name__ == "__main__":
    main()