import numpy as np
import pandas as pd
import itertools
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.spatial.distance import euclidean
from typing import List, Tuple, Optional
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination.default import DefaultMultiObjectiveTermination

class FluorophoreSelectionProblem(Problem):
    def __init__(self, crosstalk_matrix, df, num_fluorophores):
        super().__init__(n_var=len(crosstalk_matrix), 
                         n_obj=2, 
                         n_constr=1, 
                         xl=0, 
                         xu=1, 
                         type_var=np.bool_)
        
        self.crosstalk_matrix = crosstalk_matrix
        self.df = df
        self.num_fluorophores = num_fluorophores
        self.fluorophore_list = list(crosstalk_matrix.index)

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.zeros((X.shape[0], 2))  # Two objectives: Crosstalk, Spectral Overlap
        G = np.zeros((X.shape[0], 1))  # Constraint: Must select exact number of fluorophores

        for i, row in enumerate(X):
            selected_fluorophores = [self.fluorophore_list[j] for j in range(len(row)) if row[j] > 0.5]

            if len(selected_fluorophores) != self.num_fluorophores:
                F[i, :] = np.inf  # Penalize invalid solutions
                G[i, 0] = abs(len(selected_fluorophores) - self.num_fluorophores)
                continue

            subset_matrix = self.crosstalk_matrix.loc[selected_fluorophores, selected_fluorophores]
            F[i, 0] = np.mean(subset_matrix.to_numpy())  # Objective 1: Crosstalk

            F[i, 1] = compute_spectral_separation(self.df, selected_fluorophores)  # Objective 2: Spectral Overlap

        out["F"] = F
        out["G"] = G

def find_optimal_fluorophore_set_MOO(crosstalk_matrix: pd.DataFrame, 
                                 df: pd.DataFrame, 
                                 num_fluorophores: int, 
                                 population_size=100):
    """
    Finds a Pareto front of fluorophore combinations that minimize crosstalk and spectral overlap.

    Args:
        crosstalk_matrix (pd.DataFrame): Crosstalk matrix between fluorophores.
        df (pd.DataFrame): DataFrame containing excitation and emission spectra.
        num_fluorophores (int): Number of fluorophores to select.
        population_size (int): Number of solutions per generation.
        generations (int): Number of generations to evolve.

    Returns:
        List of Pareto-optimal fluorophore combinations.
    """
    problem = FluorophoreSelectionProblem(crosstalk_matrix, df, num_fluorophores)

    algorithm = NSGA2(
        pop_size=population_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    result = minimize(
        problem,
        algorithm,
        termination= DefaultMultiObjectiveTermination(),
        seed=1,
        verbose=True
    )

    # Extract Pareto front solutions
    pareto_solutions = []
    for row in result.X:
        selected_fluorophores = [crosstalk_matrix.index[i] for i in range(len(row)) if row[i] > 0.5]
        pareto_solutions.append(selected_fluorophores)

    return pareto_solutions

def wavelength_to_rgb(wavelength):
    """
    Converts a given wavelength in nm to an approximate RGB color.
    Uses a visible light spectrum approximation.
    """
    if 380 <= wavelength < 440:
        r, g, b = -(wavelength - 440) / (440 - 380), 0.0, 1.0
    elif 440 <= wavelength < 490:
        r, g, b = 0.0, (wavelength - 440) / (490 - 440), 1.0
    elif 490 <= wavelength < 510:
        r, g, b = 0.0, 1.0, -(wavelength - 510) / (510 - 490)
    elif 510 <= wavelength < 580:
        r, g, b = (wavelength - 510) / (580 - 510), 1.0, 0.0
    elif 580 <= wavelength < 645:
        r, g, b = 1.0, -(wavelength - 645) / (645 - 580), 0.0
    elif 645 <= wavelength < 700:
        r, g, b = 1.0, 0.0, 0.0
    else:
        r, g, b = 0.0, 0.0, 0.0  # Out of visible range

    return f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"

def compute_crosstalk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the spectral crosstalk matrix for a given fluorescence dataset.
    
    Args:
        df (pd.DataFrame): DataFrame containing wavelength, excitation, and emission data.
    
    Returns:
        pd.DataFrame: Crosstalk matrix where rows represent emitting fluorophores
                      and columns represent absorbing fluorophores.
    """
    fluorophores = [col.split()[0] for col in df.columns[1::2]]  
    num_fluorophores = len(fluorophores)
    crosstalk_matrix = np.zeros((num_fluorophores, num_fluorophores))

    wavelengths = df.iloc[:, 0].values  # Extract wavelength column for integration

    for i, f1 in enumerate(fluorophores):
        em_col = f"{f1} EM"
        if em_col not in df.columns:
            continue
        total_emission = np.trapezoid(df[em_col].values, wavelengths)

        for j, f2 in enumerate(fluorophores):
            if f1 == f2:
                continue
            ex_col = f"{f2} EX"
            if ex_col in df.columns:
                crosstalk = np.trapezoid(df[em_col].values * df[ex_col].values, wavelengths)
                crosstalk_matrix[i, j] = crosstalk / total_emission if total_emission > 0 else 0

    return pd.DataFrame(crosstalk_matrix, index=fluorophores, columns=fluorophores)

def compute_spectral_separation(df: pd.DataFrame, selected_fluorophores: List[str]) -> float:
    """
    Computes a spectral separation score based on excitation and emission overlap.
    Lower values indicate better separation.
    """
    total_overlap = 0
    
    wavelengths = df.iloc[:, 0].values

    for f1, f2 in itertools.combinations(selected_fluorophores, 2):
        ex_col1, ex_col2 = f"{f1} EX", f"{f2} EX"
        em_col1, em_col2 = f"{f1} EM", f"{f2} EM"
        ex1 = df[ex_col1].values
        ex2 = df[ex_col2].values
        em1 = df[em_col1].values
        em2 = df[em_col2].values
        
        total_ex1 =  np.trapezoid(ex1, wavelengths)
        total_ex2 =  np.trapezoid(em2, wavelengths)
        total_em1 =  np.trapezoid(em1, wavelengths)
        total_em2 =  np.trapezoid(em2, wavelengths)
        
        excitation_overlap = np.trapezoid(ex1*ex2, wavelengths)/(total_ex1+total_ex2)
        emission_overlap = np.trapezoid(em1*em2, wavelengths)/(total_em1+total_em2)
        total_overlap += excitation_overlap + emission_overlap

    return total_overlap  # Lower is better


def find_optimal_fluorophore_set(crosstalk_matrix: pd.DataFrame, 
                                 df: pd.DataFrame, 
                                 num_fluorophores: int) -> Tuple[Optional[List[str]], pd.DataFrame, float, float]:
    """
    Identifies the optimal set of fluorophores that ensure spectral separation 
    and then minimize crosstalk.

    Args:
        crosstalk_matrix (pd.DataFrame): Crosstalk matrix.
        df (pd.DataFrame): DataFrame containing excitation/emission spectra.
        num_fluorophores (int): Number of fluorophores to select.

    Returns:
        Tuple: (Best fluorophore combination, Crosstalk matrix subset, Min crosstalk, Min spectral overlap)
    """
    fluorophore_list = list(crosstalk_matrix.index)
    best_combination = None
    min_crosstalk = float('inf')
    min_spectral_overlap = float('inf')

    for combination in itertools.combinations(fluorophore_list, num_fluorophores):
        subset_matrix = crosstalk_matrix.loc[list(combination), list(combination)]
        total_crosstalk = np.mean(subset_matrix.to_numpy())

        # Compute spectral separation (penalty for overlap)
        spectral_separation_score = compute_spectral_separation(df, list(combination))

        # Ensure spectral separation first
        if spectral_separation_score < min_spectral_overlap:
            min_spectral_overlap = spectral_separation_score
            best_combination = combination
            best_combination_matrix = subset_matrix
            min_crosstalk = total_crosstalk

        # If spectral separation is identical, prefer lower crosstalk
        elif spectral_separation_score == min_spectral_overlap and total_crosstalk < min_crosstalk:
            min_crosstalk = total_crosstalk
            best_combination = combination
            best_combination_matrix = subset_matrix

    return best_combination, best_combination_matrix, min_crosstalk, min_spectral_overlap

def propagate_excitation(crosstalk_matrix, excitation_vector, max_iterations=10, tolerance=1e-4):
    
    # Initialize previous excitation to be the same as the current
    previous_excitation_vector = np.zeros_like(excitation_vector)
    
    # Iteratively propagate the excitation
    for _ in range(max_iterations):
        # Calculate the emission by multiplying with the crosstalk matrix
        emission_vector = np.dot(crosstalk_matrix, excitation_vector)
        
        # If the change between previous and current is below tolerance, stop
        if np.allclose(excitation_vector, previous_excitation_vector, atol=tolerance):
            break
        
        previous_excitation_vector = excitation_vector
    
        excitation_vector += emission_vector # suboptimal method, leads to divergence is excitation is large
    
    return excitation_vector
    
def compute_expected_emission(df: pd.DataFrame, fluorophores: List[str], excitation_spectrum: np.ndarray) -> np.ndarray:
    """
    Computes the expected total emission spectrum when a specific excitation light is applied.
    
    Args:
        df (pd.DataFrame): DataFrame containing excitation and emission spectra.
        fluorophores (List[str]): Selected fluorophores.
        excitation_spectrum (np.ndarray): Intensity of the excitation light at each wavelength.
    
    Returns:
        np.ndarray: The total expected emission spectrum.
    """
    wavelengths = df.iloc[:, 0].values
    total_emission = np.zeros_like(wavelengths, dtype=np.float64)

    for fluorophore in fluorophores:
        ex_col = f"{fluorophore} EX"
        em_col = f"{fluorophore} EM"

        if ex_col in df.columns and em_col in df.columns:
            excitation_response = df[ex_col].values * excitation_spectrum  # Effective excitation
            absorbed_light = np.trapezoid(excitation_response, wavelengths)  # Total absorbed energy
            
            if absorbed_light > 0:
                normalized_emission = df[em_col].values * (absorbed_light / np.trapezoid(df[em_col].values, wavelengths))
                total_emission += normalized_emission
    
    return total_emission

def compute_expected_emission_crosstalk(df: pd.DataFrame, fluorophores: List[str], excitation_spectrum: np.ndarray, crosstalk_matrix) -> np.ndarray:
    """
    Computes the expected total emission spectrum when a specific excitation light is applied.
    
    Args:
        df (pd.DataFrame): DataFrame containing excitation and emission spectra.
        fluorophores (List[str]): Selected fluorophores.
        excitation_spectrum (np.ndarray): Intensity of the excitation light at each wavelength.
    
    Returns:
        np.ndarray: The total expected emission spectrum.
    """
    
    wavelengths = df.iloc[:, 0].values
    total_emission = np.zeros_like(wavelengths, dtype=np.float64)
    
    excitation_vector = np.zeros((crosstalk_matrix.shape[1],1))
    for i,fluorophore in enumerate(fluorophores):
        ex_col = f"{fluorophore} EX"
        
        if ex_col in df.columns:
            total_excitation = np.trapezoid(df[ex_col].values, wavelengths)
            excitation_response = np.trapezoid(df[ex_col].values * excitation_spectrum, wavelengths) / total_excitation
        
        excitation_vector[i] = excitation_response
    
    emission_vector = propagate_excitation(crosstalk_matrix, excitation_vector, max_iterations=10, tolerance=1e-4)
    
    for i,fluorophore in enumerate(fluorophores):
        em_col = f"{fluorophore} EM"
        
        normalized_emission = df[em_col].values * excitation_vector[i]
        total_emission += normalized_emission
    
    return total_emission

def gaussian_excitation(wavelengths, peak, width):
    return np.exp(-((wavelengths - peak) ** 2) / (2 * width ** 2))

def band_excitation(wavelengths, center, width):
    return np.where((wavelengths >= center - width/2) & (wavelengths <= center + width/2), 1, 0).astype(np.float64)

def compute_spectral_overlap_regions(df, fluorophores):
    """
    Computes the spectral regions where excitation and emission overlaps occur.

    Args:
        df (pd.DataFrame): Fluorophore spectral data.
        fluorophores (List[str]): Selected fluorophores.

    Returns:
        Tuple[List[Tuple[List[float], List[float], str]],  # Excitation overlaps
              List[Tuple[List[float], List[float], str]]]  # Emission overlaps
    """
    wavelengths = df.iloc[:, 0].values
    excitation_overlap_regions = []
    emission_overlap_regions = []

    # Iterate over all fluorophore pairs
    for f1, f2 in itertools.combinations(fluorophores, 2):
        ex_col1 = f"{f1} EX"
        ex_col2 = f"{f2} EX"
        em_col1 = f"{f1} EM"
        em_col2 = f"{f2} EM"

        # Compute excitation overlap
        if ex_col1 in df.columns and ex_col2 in df.columns:
            ex_overlap = df[ex_col1] * df[ex_col2]
            if ex_overlap.sum() > 0:  # Ensure there's an actual overlap
                excitation_overlap_regions.append((wavelengths, ex_overlap, f"{f1}-{f2}"))

        # Compute emission overlap
        if em_col1 in df.columns and em_col2 in df.columns:
            em_overlap = df[em_col1] * df[em_col2]
            if em_overlap.sum() > 0:
                emission_overlap_regions.append((wavelengths, em_overlap, f"{f1}-{f2}"))

    return excitation_overlap_regions, emission_overlap_regions

def compute_crosstalk_regions(df, fluorophores):
    """
    Computes the spectral regions where crosstalk occurs.
    
    Args:
        df (pd.DataFrame): Fluorophore spectral data.
        crosstalk_matrix (pd.DataFrame): Crosstalk values.
        fluorophores (List[str]): Selected fluorophores.

    Returns:
        List[Tuple[List[float], List[float], str]]: Crosstalk region data with x, y, and label.
    """
    wavelengths = df.iloc[:, 0].values
    crosstalk_regions = []

    for f1, f2 in itertools.product(fluorophores, repeat=2):  # Include all n × n interactions
        if f1 != f2:  # Avoid self-crosstalk
            em_col = f"{f1} EM"
            ex_col = f"{f2} EX"

            if em_col in df.columns and ex_col in df.columns:
                # Compute spectral overlap
                overlap_intensity = df[em_col] * df[ex_col]
                if overlap_intensity.sum() > 0:  # Ensure overlap exists
                    crosstalk_regions.append((wavelengths, overlap_intensity, f"{f1} → {f2}"))

    return crosstalk_regions

def plot_spectral_overlap_interactive(df, crosstalk_matrix, fluorophores):
    """
    Plots fluorophore excitation & emission spectra with highlighted overlap regions using Plotly.
    
    Args:
        df (pd.DataFrame): Spectral data.
        crosstalk_matrix (pd.DataFrame): Crosstalk values.
        fluorophores (List[str]): Selected fluorophores.
    """
    wavelengths = df.iloc[:, 0].values
    excitation_overlap_regions, emission_overlap_regions = compute_spectral_overlap_regions(df, fluorophores)
    crosstalk_regions = compute_crosstalk_regions(df, fluorophores)

    fluorophore_colors,_ = get_fluorophore_colors(df, fluorophores)
    fig = go.Figure()

    # Plot individual spectra with assigned colors
    for fluorophore in fluorophores:
        color = fluorophore_colors[fluorophore]  # Get fluorophore color
        em_col, ex_col = f"{fluorophore} EM", f"{fluorophore} EX"

        if em_col in df.columns:
            fig.add_trace(go.Scatter(
                x=wavelengths, y=df[em_col], mode='lines',
                name=f"{fluorophore} Emission", line=dict(width=2, color=color)
            ))
        if ex_col in df.columns:
            fig.add_trace(go.Scatter(
                x=wavelengths, y=df[ex_col], mode='lines',
                name=f"{fluorophore} Excitation", line=dict(width=2, dash='dash', color=color)
            ))

    # Highlight excitation overlap regions
    for wavelengths, intensity, label in excitation_overlap_regions:
        fig.add_trace(go.Scatter(
            x=wavelengths, y=intensity, fill='tozeroy', mode='none',
            name=f"Exc. Overlap: {label}", fillcolor='rgba(255, 165, 0, 0.3)'
        ))

    # Highlight emission overlap regions
    for wavelengths, intensity, label in emission_overlap_regions:
        fig.add_trace(go.Scatter(
            x=wavelengths, y=intensity, fill='tozeroy', mode='none',
            name=f"Em. Overlap: {label}", fillcolor='rgba(0, 255, 255, 0.3)'
        ))

    # Highlight crosstalk regions
    for wavelengths, intensity, label in crosstalk_regions:
        fig.add_trace(go.Scatter(
            x=wavelengths, y=intensity, fill='tozeroy', mode='none',
            name=f"Crosstalk: {label}", fillcolor='rgba(255, 0, 0, 0.3)'
        ))

    # Update layout with overlap percentage info
    fig.update_layout(
        title=f"Fluorophore Spectra",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Intensity",
        template="plotly_dark",
        hovermode="x unified"
    )

    fig.show()
    
def get_fluorophore_colors(df, fluorophores):
    """
    Determines the color for each fluorophore based on its peak emission wavelength.

    Args:
        df (pd.DataFrame): Fluorophore spectral data.
        fluorophores (List[str]): Selected fluorophores.

    Returns:
        Dict[str, str]: Fluorophore-to-color mapping.
    """
    fluorophore_colors = {}
    peak_excitation_wavelengths = {}
    wavelengths = df.iloc[:, 0].values  # Wavelength column

    for fluorophore in fluorophores:
        em_col = f"{fluorophore} EM"
        ex_col = f"{fluorophore} EX"
        if em_col in df.columns:
            peak_wavelength = wavelengths[df[em_col].idxmax()]  # Find peak emission wavelength
            fluorophore_colors[fluorophore] = wavelength_to_rgb(peak_wavelength)
            
        if ex_col in df.columns:
            peak_excitation_wavelength = wavelengths[df[ex_col].idxmax()]  # Find peak emission wavelength
            peak_excitation_wavelengths[fluorophore] = peak_excitation_wavelength

    return fluorophore_colors, peak_excitation_wavelengths

def plot_expected_emission_interactive(wavelengths: np.ndarray, expected_emission: np.ndarray):
    """
    Creates an interactive plot of the expected emission spectrum using Plotly.
    
    Args:
        wavelengths (np.ndarray): Wavelength values.
        expected_emission (np.ndarray): Computed emission intensity.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=wavelengths, y=expected_emission, mode='lines', 
                             name="Expected Emission", line=dict(width=2, color="red")))

    fig.update_layout(
        title="Expected Emission Spectrum",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Intensity",
        template="plotly_dark",
        hovermode="x unified"
    )

    fig.show()
    
def plot_simulated_emission_interactive(df: pd.DataFrame, fluorophores: list, lights: list, wavelengths: np.ndarray, crosstalk_matrix: np.ndarray):
    """
    Simulates and plots expected emission spectra when exciting fluorophores with given light sources using Plotly.

    Args:
        df (pd.DataFrame): Spectral data with "Excitation" and "Emission" spectra.
        fluorophores (list): Selected fluorophores to simulate.
        lights (list): List of (center_wavelength, bandwidth) for each light source.
        wavelengths (np.ndarray): Array of wavelengths for evaluation.
    """
    fig = go.Figure()
    
    fluorophore_colors, peak_excitation_wavelengths = get_fluorophore_colors(df, fluorophores)  # Use the same color scheme as your other plots

    if not lights:
        lights = [(wavelength.astype(np.float64),10) for wavelength in peak_excitation_wavelengths.values()]

    i = 0
    for center, width in lights:
        excitation_light = band_excitation(wavelengths,center,width)
        # excitation_light = gaussian_excitation(wavelengths, peak=488, width=10)
        excitation_color = wavelength_to_rgb(center)
        
        # Add excitation light as a dashed line
        fig.add_trace(go.Scatter(
            x=wavelengths, y=excitation_light, mode='lines',
            name=f"ExcitationLight:{center},{width})", line=dict(width=2, dash='dash', color=excitation_color)
        ))
        
        for f in fluorophores:
            # expected_emission = compute_expected_emission_crosstalk(df, best_set, excitation_light)
            expected_emission = compute_expected_emission_crosstalk(df, best_set, excitation_light, crosstalk_matrix)
            excitation_light += expected_emission # not ideal solution, may need to loop over multiple times?
            
        # Add expected emission as solid line
        fig.add_trace(go.Scatter(
            x=wavelengths, y=expected_emission, mode='lines',
            name=f"Total Emission (ExcitationLight:{center},{width})", line=dict(width=2, color=excitation_color)
        ))
        
        i+=1

    # # Overlay excitation bands
    # for i, (center, width) in enumerate(lights):
    #     fig.add_trace(go.Scatter(
    #         x=[center - width / 2, center + width / 2, center + width / 2, center - width / 2],
    #         y=[0, 0, max(total_emission) * 1.1, max(total_emission) * 1.1],
    #         fill="toself",
    #         name=f"Excitation {center} nm",
    #         fillcolor="rgba(255, 255, 0, 0.3)",
    #         line=dict(color="rgba(255, 255, 0, 0.3)")
    #     ))

    fig.update_layout(
        title="Simulated Emission Spectrum Under Multi-Source Excitation",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Intensity",
        template="plotly_dark",
        hovermode="x unified"
    )

    fig.show()

if __name__ == "__main__":
    file_path = "ATTO465_HEX_TR_CY5_5.csv"

    df = pd.read_csv(file_path)
    df.fillna(0, inplace=True)  # Replace NaNs with 0

    crosstalk_matrix = compute_crosstalk(df)

    num_fluorophores = 4
    
    # pareto_solutions = find_optimal_fluorophore_set_MOO(crosstalk_matrix, 
    #                              df, 
    #                              num_fluorophores, 
    #                              population_size=10)
    
    # print(pareto_solutions)
    
    best_set, best_set_matrix, min_crosstalk, min_overlap = find_optimal_fluorophore_set(crosstalk_matrix, df, num_fluorophores)

    print("\nOptimal fluorophore combination with minimum crosstalk:")
    print(best_set)
    print("\nCrosstalk matrix:")
    print(best_set_matrix)
    print(f"Mean Crosstalk: {min_crosstalk:.5f}")
    print(f"Mean Excitation Spectral Overlap: {min_overlap:.5f}")
    
    plot_spectral_overlap_interactive(df, crosstalk_matrix, best_set)
    
    wavelengths = df.iloc[:, 0].values
    # excitation_light = gaussian_excitation(wavelengths, peak=488, width=10)
    # expected_emission = compute_expected_emission(df, best_set, excitation_light)
    # plot_expected_emission_interactive(wavelengths, expected_emission)
    
    excitation_lights = [] # ((495,10),(540,10),(580,10),(655,10))
    plot_simulated_emission_interactive(df, best_set, excitation_lights, wavelengths, best_set_matrix)