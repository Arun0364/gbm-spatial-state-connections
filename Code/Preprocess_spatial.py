# Imports
import argparse
import scanpy as sc
import pandas as pd
import numpy as np
import json
import os

# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess spatial transcriptomics data')
    parser.add_argument('data_directory', type=str)
    parser.add_argument('output_directory', type=str)
    parser.add_argument('Sample_Name', type=str)
    parser.add_argument('Sample_ID', type=str)
    return parser.parse_args()

# Preprocessing function
def preprocess_spatial_data(data_dir, output_dir, sample_id, sample_name):
    matrix_file = next(os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith("_matrix.mtx"))
    features_file = next(os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith("_features.tsv"))
    barcodes_file = next(os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith("_barcodes.tsv"))
    spatial_dir = os.path.join(data_dir, "spatial")
    pos_file = next(os.path.join(spatial_dir, f) for f in os.listdir(spatial_dir) if f == "tissue_positions.csv")
    scale_factors_file = next(os.path.join(spatial_dir, f) for f in os.listdir(spatial_dir) if f == "scalefactors_json.json")
    hires_img_file = next(os.path.join(spatial_dir, f) for f in os.listdir(spatial_dir) if f == "tissue_hires_image.png")
    lowres_img_file = next(os.path.join(spatial_dir, f) for f in os.listdir(spatial_dir) if f == "tissue_lowres_image.png")
    
    X = sc.read_mtx(matrix_file)
    barcodes = pd.read_csv(barcodes_file, header=None, sep="\t")
    barcodes.columns = ["barcode"]
    
    features = pd.read_csv(features_file, header=None, sep="\t")
    features.columns = ["gene_id", "gene_name"] + [f"col_{i}" for i in range(features.shape[1] - 2)]
    
    pos = pd.read_csv(pos_file)
    pos_in = pos[pos["in_tissue"].astype(int) == 1].copy()
    pos_in = pos_in.set_index("barcode").loc[barcodes["barcode"]].reset_index()

    scaleFactors = json.load(open(scale_factors_file, "r"))
    
    aData = sc.AnnData(X.T)
    aData.obs_names = barcodes["barcode"].astype(str).values
    aData.var["gene_id"] = features["gene_id"].astype(str).values
    aData.var_names = features["gene_name"].astype(str).values
    aData.var_names_make_unique()

    aData.obs["sample_ID"] = sample_id
    aData.obs["sample_name"] = sample_name
    aData.obs["in_tissue"] = pos_in["in_tissue"].astype(int).values
    aData.obs["array_row"] = pos_in["array_row"].astype(int).values
    aData.obs["array_col"] = pos_in["array_col"].astype(int).values
    aData.obs["pxl_row_in_fullres"] = pos_in["pxl_row_in_fullres"].astype(int).values
    aData.obs["pxl_col_in_fullres"] = pos_in["pxl_col_in_fullres"].astype(int).values
    aData.obsm["spatial"] = np.vstack([
        aData.obs["pxl_col_in_fullres"].to_numpy(),
        aData.obs["pxl_row_in_fullres"].to_numpy()
    ]).T
    libraryID = aData.obs["sample_ID"].iloc[0]
    aData.uns["spatial"] = {
        libraryID: {
            "scalefactors": scaleFactors,
            "images": {},
            "metadata": {"chemistry_description": "Visium", "source": "GEO"}
            }
    }
    try:
        import matplotlib.image as mpimg
        if hires_img_file and os.path.exists(hires_img_file):
            aData.uns["spatial"][libraryID]["images"]["hires"] = mpimg.imread(hires_img_file)
        if lowres_img_file and os.path.exists(lowres_img_file):
            aData.uns["spatial"][libraryID]["images"]["lowres"] = mpimg.imread(lowres_img_file)
    except Exception as e:
        print("Warning: could not load tissue images:", e)

    aData.var["mt"] = aData.var_names.str.startswith(("MT-", "mt-"))

    aData.layers["counts"] = aData.X.copy()

    out_file = os.path.join(output_dir, f"{sample_name}_preprocessed_spatial_data.h5ad")
    aData.write_h5ad(out_file)
    print(f"Preprocessed data saved to {out_file}")

# Main function
def main():
    args = parse_args()
    preprocess_spatial_data(args.data_directory, args.output_directory, args.Sample_ID, args.Sample_Name)

if __name__ == "__main__":
    main()