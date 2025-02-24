import numpy as np
import pandas as pd
import itertools

def compute_crosstalk(df):
    fluorophores = [col.split()[0] for col in df.columns[1::2]]  # Extract fluorophore names
    crosstalk_matrix = pd.DataFrame(0.0, index=fluorophores, columns=fluorophores, dtype=np.float64)

    for f1, f2 in itertools.product(fluorophores, repeat=2):
        if f1 != f2:
            em_col = f"{f1} EM"
            ex_col = f"{f2} EX"
            if em_col in df.columns and ex_col in df.columns:
                total_emission = np.trapezoid(df[em_col], df.iloc[:, 0])
                crosstalk = np.trapezoid(df[em_col] * df[ex_col], df.iloc[:, 0])

                crosstalk_matrix.loc[f1, f2] = crosstalk/ total_emission

    return crosstalk_matrix

def compute_spectral_overlap(df, fluorophores):
    wavelengths = df.iloc[:, 0]
    total_overlap = 0.0

    for f1, f2 in itertools.combinations(fluorophores, 2):
        em_col1 = f"{f1} EM"
        em_col2 = f"{f2} EM"
        if em_col1 in df.columns and em_col2 in df.columns:
            # Compute emission spectral overlap
            total_emission_1 = np.trapezoid(df[em_col1], df.iloc[:, 0])
            total_emission_2 = np.trapezoid(df[em_col2], df.iloc[:, 0])
            total_overlap += np.trapezoid(df[em_col1] * df[em_col2], wavelengths) / max(total_emission_1,total_emission_2) # normalize using larger total emission

    return total_overlap/ len(fluorophores)

def find_optimal_fluorophore_set(crosstalk_matrix, df, num_fluorophores):

    fluorophore_list = list(crosstalk_matrix.index)
    best_combination = None
    min_crosstalk = float('inf')
    min_spectral_overlap = float('inf')

    for combination in itertools.combinations(fluorophore_list, num_fluorophores):
        subset_matrix = crosstalk_matrix.loc[combination, combination]
        total_crosstalk = subset_matrix.to_numpy().mean()  # Sum of crosstalk values

        # Compute spectral overlap separately
        spectral_overlap = compute_spectral_overlap(df, combination)

        # Select set with minimal crosstalk, then minimal spectral overlap
        if total_crosstalk < min_crosstalk or (total_crosstalk == min_crosstalk and spectral_overlap < min_spectral_overlap):
            min_crosstalk = total_crosstalk
            min_spectral_overlap = spectral_overlap
            best_combination = combination

    return best_combination, min_crosstalk, min_spectral_overlap

if __name__ == "__main__":
    file_path = "SpectralCrosstalkCalculator\FPbase_Spectra_All.csv"

    df = pd.read_csv(file_path)
    df.fillna(0, inplace=True)  # Replace NaNs with 0

    crosstalk_matrix = compute_crosstalk(df)

    num_fluorophores = 6
    best_set, min_crosstalk, min_overlap = find_optimal_fluorophore_set(crosstalk_matrix, df, num_fluorophores)

    print("\nOptimal fluorophore combination with minimum crosstalk:")
    print(best_set)
    print(f"Total Crosstalk: {min_crosstalk:.5f}")
    print(f"Total Emission Spectral Overlap: {min_overlap:.5f}")