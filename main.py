import argparse
from TableCluster.tableClustering import silm_clustering, files_columns_running, files_hierarchyInference
from RelationshipSearch.SearchRelationship import  relationshipDiscovery
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="starmie")  # Valerie starmie
    parser.add_argument("--dataset", type=str, default="TabFact")  # WDC TabFact
    parser.add_argument("--logdir", type=str, default="model/")
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--projector", type=int, default=768)
    parser.add_argument("--augment_op", type=str, default='drop_num_col')
    parser.add_argument("--strict", dest="strict", action="store_true")

    parser.add_argument("--embed", type=str, default='bert')

    parser.add_argument("--embedMethod", type=str, default='')
    parser.add_argument("--save_model", dest="save_model", action="store_true", default=True)
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--slice_start", type=int, default=1)
    parser.add_argument("--slice_stop", type=int, default=1)

    parser.add_argument("--intervalSlice", type=int, default=20)
    parser.add_argument("--delta", type=int, default=0.3)

    parser.add_argument("--column", dest="column", action="store_true", default=True)
    # single-column mode without table context
    parser.add_argument("--subjectCol", dest="subjectCol", action="store_true")
    # row / column-ordered for preprocessing
    parser.add_argument("--table_order", type=str, default='column')  # column
    parser.add_argument("--phaseTest", dest="phaseTest", action="store_true")
    # for sampling
    parser.add_argument("--sample_meth", type=str, default='head')  # head tfidfrow
    # mlflow tag

    hp = parser.parse_args()
    if hp.step == 1:
        silm_clustering(hp)
    if hp.step == 2:
        files_columns_running(hp)
    if hp.step ==3:
        files_hierarchyInference(hp)
    if hp.step ==4:
        relationshipDiscovery(hp)
