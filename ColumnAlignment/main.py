import argparse
from SimpleMatch import SimpleColumnMatch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="M2")  # M1 M2 M3
    parser.add_argument("--dataset", type=str, default="open_data") # ChEMBL Magellan OpenData TPC-DI Wikidata
    parser.add_argument("--eval_path", type=str, default="data/")  # ChEMBL Magellan OpenData TPC-DI Wikidata
    parser.add_argument("--fine_tune",dest="fine_tune", action="store_true", default=True)
    parser.add_argument("--logdir", type=str, default="model/")
    parser.add_argument("--output_dir", type=str, default="model/")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--early_stop", type=bool, default=False)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--lm", type=str, default='roberta')
    parser.add_argument("--positive_op", type=str, default='random')
    parser.add_argument("--save_model", dest="save_model", action="store_true", default=True)

    hp = parser.parse_args()


    # Change the data paths to where the benchmarks are stored
    path = 'ValentineDatasets/%s/Train/' % hp.dataset
    if hp.method == "M1" or hp.method == "M2":
        match = SimpleColumnMatch(hp.eval_path,hp.method)
        score = match.SimpleMatch(0.5)
        print(score)





