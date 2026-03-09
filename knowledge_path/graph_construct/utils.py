import pandas as pd
import glob


def make_relations_csv(csv_dir: str) -> pd.DataFrame:
    """
    Reads multiple CSV files in a directory and returns a combined DataFrame.

    Args:
        file_path (str): The path to the CSV file.
    Returns:
        pd.DataFrame: The DataFrame containing the CSV data.
    """
    csv_files = glob.glob(f"{csv_dir}/*.csv")
    dfs = []
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        df_relations = df[['reference_entity', 'related_entity', 'relation_name']]
        dfs.append(df_relations)
    #save csv w
    df_combined = pd.concat(dfs, ignore_index=True)
    #change column names
    # df_combined.columns = [':START_ID', ':END_ID', ':TYPE', 'RELA']
    df_combined.to_csv(f"{csv_dir}/relations.csv", index=False)
    return df_combined

if __name__ == "__main__":
    csv_dir = "/home/nadun/wd/vlm_histo/data/wsi_bench/custom_kg"
    make_relations_csv(csv_dir)

