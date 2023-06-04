import argparse
import os.path
from ColumnMatch import ColumnMatch
from SimpleMatch import SimpleColumnMatch
from metrics import Ground_truth

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="M1")  # M1 M2 M3
    parser.add_argument("--dataset", type=str, default="Wikidata")  # ChEMBL Magellan OpenData TPC-DI Wikidata
    parser.add_argument("--ground_truth_path", type=str, default="data/data_mapping.json")  #
    parser.add_argument("--eval_path", type=str, default="data/")  # data/
    parser.add_argument("--fine_tune", dest="fine_tune", action="store_true", default=False)
    parser.add_argument("--logdir", type=str, default="model/")
    parser.add_argument("--output_dir", type=str, default="model/")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--early_stop", type=bool, default=False)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--lm", type=str, default='roberta')
    parser.add_argument("--positive_op", type=str, default='random')
    parser.add_argument("--save_model", dest="save_model", action="store_true", default=False)

    hp = parser.parse_args()

    # Change the data paths to where the benchmarks are stored
    path = 'ValentineDatasets/%s/Train/' % hp.dataset

    if hp.method == "M1" or hp.method == "M2":
        match = SimpleColumnMatch(hp.eval_path, hp.method)
        score = match.SimpleMatch(0.55)
        Ground_truth(hp.eval_path,hp.ground_truth_path, score, hp.eval_path + "/results/", hp.method)
    if hp.method == "M3":
        current_dir = os.getcwd()
        parent_dir = os.path.dirname(current_dir)
        absolute_path = parent_dir + "/" + path
        print(absolute_path)
        #M3_classifier = ColumnMatchingClassifier.from_hp(absolute_path,hp)
        M3 = ColumnMatch(absolute_path, hp)
        score = M3.score(hp, 0)
        print(score)
        Ground_truth(hp.eval_path,hp.ground_truth_path, score, hp.eval_path + "/results/", hp.method)
