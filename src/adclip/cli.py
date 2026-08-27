"""
CLI wrapper for query and continuing training

"""
import argparse
import sys
import pandas as pd

from .api import ADCLIP


def _print_and_save(df: pd.DataFrame, out: str):
    print(df.to_string(index=False))
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")


def _load_substrates_csv(path):
    df = pd.read_csv(path)
    return dict(zip(df["name"], df["smiles"]))


def cmd_query_adomain(args):
    model = ADCLIP.load(checkpoint=args.checkpoint, device=args.device)
    if args.substrates:
        substrate_smiles = _load_substrates_csv(args.substrates)
    else:
        substrate_smiles = None
    df = model.query_adomain(args.fasta, substrate_smiles=substrate_smiles,
                              pool=args.alignment_context, top_k=args.top_k, threads=args.threads)
    _print_and_save(df, args.out)


def cmd_query_substrate(args):
    model = ADCLIP.load(checkpoint=args.checkpoint, device=args.device)
    df = model.query_substrate(args.smiles, corpus_fasta=args.corpus_fasta,
                                   pool=args.pool, top_k=args.top_k, threads=args.threads)
    _print_and_save(df, args.out)


def cmd_cpt(args):
    model = ADCLIP.load(checkpoint=args.checkpoint, device=args.device)
    model.continue_pretrain(
        args.pairs, lr=args.lr, epochs=args.epochs, patience=args.patience,
        l_atp=args.l_atp, l_prop=args.l_prop, batch_size=args.batch_size,
        pool=args.pool, threads=args.threads, val_pairs_csv=args.val_pairs, val_split=args.val_split,
        out_checkpoint=args.out)


def build_parser():
    p = argparse.ArgumentParser(prog="adclip")
    sub = p.add_subparsers(dest="command", required=True)

    qa = sub.add_parser("query_adomain", help="A-domain sequence(s) -> ranked substrates")
    qa.add_argument("--fasta", required=True, help="FASTA path")
    qa.add_argument("--substrates", help="CSV with columns name,smiles — the pool to rank "
                                          "against (default: bundled 43-substrate)")
    qa.add_argument("--alignment_context", default="training",
                     help='alignment context for the new sequences: "training" (the full bundled '
                          'corpus) or a path to an unaligned FASTA')
    qa.add_argument("--checkpoint", default="complete", help="baseline | complete | path to a .pt file")
    qa.add_argument("--top_k", type=int, default=None)
    qa.add_argument("--threads", type=int, default=4, help="MUSCLE alignment threads")
    qa.add_argument("--out", default="query_adomain_results.csv", help="write results to CSV (default: %(default)s)")
    qa.add_argument("--device", default="cpu")
    qa.set_defaults(func=cmd_query_adomain)

    qs = sub.add_parser("query_substrate", help="substrate SMILES -> ranked A-domains")
    qs.add_argument("--smiles", required=True)
    qs.add_argument("--corpus_fasta", default=None,
                     help="raw/unaligned candidate A-domains (default: bundled training corpus)")
    qs.add_argument("--pool", default="training",
                     help='alignment pool for --corpus-fasta (ignored for the default corpus)')
    qs.add_argument("--checkpoint", default="complete")
    qs.add_argument("--top_k", type=int, default=None)
    qs.add_argument("--threads", type=int, default=4,
                     help="MUSCLE alignment threads (only used when --corpus_fasta is given)")
    qs.add_argument("--out", default="query_substrate_results.csv", help="write results to CSV (default: %(default)s)")
    qs.add_argument("--device", default="cpu")
    qs.set_defaults(func=cmd_query_substrate)

    cpt = sub.add_parser("cpt", help="continual pretraining on new (A-domain, substrate) pairs")
    cpt.add_argument("--pairs", required=True,
                      help="CSV with column a_domain_sequence, plus smiles (for a brand-new "
                           "substrate) and/or substrate_name (for a substrate the checkpoint "
                           "already knows)")
    cpt.add_argument("--checkpoint", default="complete", help="baseline | complete | path")
    cpt.add_argument("--lr", type=float, default=1e-4)
    cpt.add_argument("--epochs", type=int, default=50)
    cpt.add_argument("--patience", type=int, default=5)
    cpt.add_argument("--l_atp", type=float, default=0.2)
    cpt.add_argument("--l_prop", type=float, default=0.3)
    cpt.add_argument("--batch_size", type=int, default=64)
    cpt.add_argument("--pool", default="training")
    cpt.add_argument("--threads", type=int, default=4,
                      help="MUSCLE alignment threads -- bump this up on a multi-core machine "
                           "for a faster full-corpus alignment")
    cpt.add_argument("--val_pairs", default=None,
                      help="CSV with the same columns as --pairs, used as the validation set "
                           "instead of an automatic split of the bundled corpus. Required if "
                           "--checkpoint has no held-out substrates (e.g. 'complete') -- there's "
                           "no untouched corpus data left to validate against automatically.")
    cpt.add_argument("--val_split", default="substrate", choices=["substrate", "row"],
                      help="only used when --val_pairs isn't given: 'substrate' (default) keeps "
                           "disjoint substrates between train/val; 'row' allows the same "
                           "substrates in both, splitting by row instead")
    cpt.add_argument("--out", required=True, help="path to save the resulting checkpoint")
    cpt.add_argument("--device", default="cpu")
    cpt.set_defaults(func=cmd_cpt)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
