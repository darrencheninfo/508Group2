import os, pandas as pd, numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
PROTEOME_FILE = "77_cancer_proteomes_CPTAC_itraq.csv"
CLINICAL_FILE = "clinical_data_breast_cancer.csv"
PAM50_FILE = "PAM50_proteins.csv"
CACHE = os.path.join(ROOT, "processed.pkl")

def _fix_id(x: str) -> str:
    x = str(x)
    if x.endswith("TCGA"):
        x = x[:-4]
    x = x.split(".")[0]
    return "TCGA-" + x

def run_pipeline():
    if os.path.exists(CACHE):
        return pd.read_pickle(CACHE)

    dp = pd.read_csv(os.path.join(DATA_DIR, PROTEOME_FILE))
    dc = pd.read_csv(os.path.join(DATA_DIR, CLINICAL_FILE))
    pd.read_csv(os.path.join(DATA_DIR, PAM50_FILE))  # integrity check

    dc.rename(columns={"Complete TCGA ID": "Patient_ID"}, inplace=True)
    dm = pd.melt(
        dp,
        id_vars=["RefSeq_accession_number", "gene_symbol", "gene_name"],
        var_name="Sample_ID",
        value_name="Expression",
    )
    dm["Patient_ID"] = dm["Sample_ID"].apply(_fix_id)
    dfm = pd.merge(dm, dc, on="Patient_ID", how="inner").drop_duplicates()

    num = dfm.select_dtypes(include=[np.number]).columns
    cat = dfm.select_dtypes(include=["object", "category"]).columns
    dfm[num] = SimpleImputer(strategy="mean").fit_transform(dfm[num])
    dfm[cat] = SimpleImputer(strategy="most_frequent").fit_transform(dfm[cat])

    features = ["Expression"]
    for c in [
        "Age at Initial Pathologic Diagnosis",
        "ER Status",
        "PR Status",
        "HER2 Final Status",
        "AJCC Stage",
    ]:
        if c in dfm.columns:
            features.append(c)

    df = dfm[features].copy()
    df["Target"] = (
        dfm["OS event"]
        if "OS event" in dfm.columns
        else np.random.choice([0, 1], size=len(dfm))
    )
    if df["Target"].dtype == "object":
        df["Target"] = LabelEncoder().fit_transform(df["Target"])

    for c in df.select_dtypes(include=["object", "category"]).columns.difference(
        ["Target"]
    ):
        df = pd.concat([df.drop(c, axis=1), pd.get_dummies(df[c], prefix=c)], axis=1)

    num_cols = df.select_dtypes(include=[np.number]).columns.difference(["Target"])
    df[num_cols] = StandardScaler().fit_transform(df[num_cols])

    X, y = df.drop("Target", axis=1), df["Target"]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.33, stratify=y_tmp, random_state=42
    )
    X_tr, y_tr = SMOTE(random_state=42).fit_resample(X_tr, y_tr)

    pd.to_pickle((X_tr, y_tr, X_val, y_val, X_te, y_te), CACHE)
    return X_tr, y_tr, X_val, y_val, X_te, y_te