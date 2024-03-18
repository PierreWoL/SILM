import argparse
from TableCluster.tableClustering import P1, P2, P3, endToEnd, baselineP1
from RelationshipSearch.SearchRelationship import relationshipDiscovery, P4

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="starmie")  #
    parser.add_argument("--dataset", type=str, default="WDC")  # WDC TabFact
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

    parser.add_argument("--embed", type=str, default='sbert')

    parser.add_argument("--embedMethod", type=str, default='')
    parser.add_argument("--baseline", dest="baseline", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true", default=True)
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--slice_start", type=int, default=0)
    parser.add_argument("--slice_stop", type=int, default=1)

    parser.add_argument("--intervalSlice", type=int, default=20)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--similarity", type=float, default=0.5)
    parser.add_argument("--clustering", type=str, default='Agglomerative')  # Agglomerative
    parser.add_argument("--iteration", type=int, default=1)  # Agglomerative

    parser.add_argument("--column", dest="column", action="store_true", default=True)
    # single-column mode without table context
    parser.add_argument("--subjectCol", dest="subjectCol", action="store_true")
    # row / column-ordered for preprocessing
    parser.add_argument("--table_order", type=str, default='column')  # column
    parser.add_argument("--phaseTest", dest="phaseTest", action="store_true")
    # for sampling
    parser.add_argument("--sample_meth", type=str, default='head')  # head tfidfrow
    parser.add_argument("--groundTruth", dest="groundTruth", action="store_true")

    parser.add_argument("--topk", type=int, default=0)

    parser.add_argument("--P1Embed", type=str, default='cl_SC8_lm_sbert_head_column_0_none_subCol.pkl')
    parser.add_argument("--P23Embed", type=str, default='cl_SC8_lm_sbert_head_column_0_header_column.pkl')
    parser.add_argument("--P4Embed", type=str, default='Pretrain_sbert_head_column_none_False.pkl')
    # TODO Needs to delete later/ or re-code
    parser.add_argument("--SelectType", type=str, default='')
    hp = parser.parse_args()
    if hp.step == 1:
        P1(hp) if hp.baseline is False else baselineP1(hp)
    elif hp.step == 2:
        P2(hp)
    elif hp.step == 3:
        P3(hp)
    elif hp.step == 4:
        relationshipDiscovery(hp)
        # P4(hp)
    elif hp.step == -1:
        endToEnd(hp)
