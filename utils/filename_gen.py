def filename_gen(alg, scaler, pca):
    filename = [alg]
    filename.append("_scaler" if scaler else "_no-scaler")
    filename.append("_pca" if pca else "_no-pca")
    filename.append(".joblib")
    return "".join(filename)
