import argparse, numpy as np
from scipy.io import loadmat
import scipy.sparse as sp

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = loadmat(args.inp)
    A = z['A']
    X = z['X']
    y = z.get('y', None)

    if sp.issparse(A): A = A.todense()
    A = np.asarray(A, dtype=np.float32)
    X = np.asarray(X, dtype=np.float32)
    if y is not None:
        y = np.asarray(y).squeeze()

    np.savez_compressed(args.out, A=A, X=X, y=y)
    print("Saved:", args.out)
