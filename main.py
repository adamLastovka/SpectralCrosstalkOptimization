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
from tabulate import tabulate
import matplotlib.pyplot as plt

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

def compute_crosstalk(wavelengths: list, df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the spectral crosstalk matrix for a given fluorescence dataset.
    
    Args:
        df (pd.DataFrame): DataFrame containing wavelength, excitation, and emission data.
    
    Returns:
        pd.DataFrame: Crosstalk matrix where rows represent emitting fluorophores
                      and columns represent absorbing fluorophores.
    """
    fluors = df.index
    num_fluorophores = len(fluors)
    crosstalk_matrix = np.zeros((num_fluorophores, num_fluorophores))

    for i, f1 in enumerate(fluors):
        em_spec = df.loc[f"{f1}","EM"] 
        total_emission = np.trapezoid(em_spec, wavelengths)

        for j, f2 in enumerate(fluors):
            if f1 == f2:
                continue
            
            ex_spec = df.loc[f"{f2}","EX"] 
            crosstalk = np.trapezoid(em_spec * ex_spec, wavelengths)
            
            crosstalk_matrix[i, j] = crosstalk / total_emission if total_emission > 0 else 0

    return pd.DataFrame(crosstalk_matrix, index=fluors, columns=fluors)

def compute_spectral_separation(wavelengths: list, df: pd.DataFrame, selected_fluorophores: List[str]) -> float:
    """
    Computes a spectral separation score based on excitation and emission overlap.
    Lower values indicate better separation.
    """
    total_overlap = 0

    for f1, f2 in itertools.combinations(selected_fluorophores, 2):
        ex1 = df.loc[f"{f1}","EX"] 
        ex2 = df.loc[f"{f2}","EX"] 
        em1 = df.loc[f"{f1}","EM"] 
        em2 = df.loc[f"{f2}","EM"] 
        
        total_ex1 =  np.trapezoid(ex1, wavelengths)
        total_ex2 =  np.trapezoid(em2, wavelengths)
        total_em1 =  np.trapezoid(em1, wavelengths)
        total_em2 =  np.trapezoid(em2, wavelengths)
        
        excitation_overlap = np.trapezoid(ex1*ex2, wavelengths)/(total_ex1+total_ex2)
        emission_overlap = np.trapezoid(em1*em2, wavelengths)/(total_em1+total_em2)
        total_overlap += excitation_overlap + emission_overlap

    return total_overlap  # Lower is better


def find_optimal_fluorophore_set(wavelengths: list,
                                 df: pd.DataFrame, crosstalk_matrix: pd.DataFrame,
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
        spectral_separation_score = compute_spectral_separation(wavelengths, df, list(combination))

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

def compute_expected_emission_crosstalk(wavelengths: list, df: pd.DataFrame, fluorophores: List[str], excitation_spectrum: np.ndarray, crosstalk_matrix) -> np.ndarray:
    """
    Computes the expected total emission spectrum when a specific excitation light is applied.
    
    Args:
        df (pd.DataFrame): DataFrame containing excitation and emission spectra.
        fluorophores (List[str]): Selected fluorophores.
        excitation_spectrum (np.ndarray): Intensity of the excitation light at each wavelength.
    
    Returns:
        np.ndarray: The total expected emission spectrum.
    """
    total_emission = np.zeros_like(wavelengths, dtype=np.float64)
    
    excitation_vector = np.zeros((crosstalk_matrix.shape[1],1))
    for i,fluor in enumerate(fluorophores):
        excitation_vector[i] = normalized_product_integral(wavelengths,df.loc[f"{fluor}", "EX"] ,excitation_spectrum)
    
    for i,fluor in enumerate(fluorophores):
        normalized_emission = df.loc[f"{fluor}", "EM"]  * excitation_vector[i]
        total_emission += normalized_emission
    
    return total_emission

def normalized_product_integral(x, y1, y2):
    """
    Returns the integral of y1*y2 normalized by the integral of y1. Assumes y1 and y2 have the same domain.
    Args:
        x (np.ndarray): Array of x values over which to numerically integrate
        y1 (np.ndarray): Array of y1 values
        y2 (np.ndarray): Array of y2 values

    Returns:
        np.float64: Normalized product integral
    """
    integral = np.trapezoid(y1, x)
    return np.trapezoid(y1 * y2, x) / integral

def normalizeArray(array: np.ndarray):
    return (array - np.min(array)) / (np.max(array) - np.min(array))

def compute_excitation_efficiency(wavelengths: list, df: pd.DataFrame, fluorophores: List[str], lights: List[str]):
    excitation_efficiency = []

    for light, fluor in zip(lights, fluorophores):
        excitation_efficiency.append(normalized_product_integral(wavelengths, light, df.loc[f"{fluor}", "EX"] ))
        
    return excitation_efficiency

def gaussian_excitation(wavelengths, peak, width):
    return np.exp(-((wavelengths - peak) ** 2) / (2 * width ** 2))

def band_excitation(wavelengths, center, width):
    return np.where((wavelengths >= center - width/2) & (wavelengths <= center + width/2), 1, 0).astype(np.float64)

def split_multiband_filter(filter: np.ndarray):
    """
    Splits single multiband array into arrays of individual bands.

    Args:
        filter_arr (_type_): 2D array [wavelengths,transmissivity]
    """
    threshold = 0.01
    
    separated_bands = []
    output_arr = []
    
    in_band = False
    current_band = []
    
    for i in range(0,np.shape(filter)[0]):
        wl = filter[i,0]
        trans = filter[i,1]
        
        if trans > threshold:
            if not in_band:
                # Start new band
                in_band = True
                
            current_band.append([wl, trans])
                
        else:
            if in_band:
                # end band and append to array
                separated_bands.append(np.asarray(current_band))
                current_band = []
                in_band = False
    
    for band in separated_bands:
        padded_arr = np.zeros((len(wavelengths),2))
        padded_arr[:,0] = wavelengths
        
        mask = np.isin(wavelengths, band[:, 0])
        padded_arr[mask,1] = band[:,1]
        
        output_arr.append(padded_arr)
         
    return output_arr

def compute_spectral_overlap_regions(wavelengths, df, fluorophores):
    """
    Computes the spectral regions where excitation and emission overlaps occur.

    Args:
        df (pd.DataFrame): Fluorophore spectral data.
        fluorophores (List[str]): Selected fluorophores.

    Returns:
        Tuple[List[Tuple[List[float], List[float], str]],  # Excitation overlaps
              List[Tuple[List[float], List[float], str]]]  # Emission overlaps
    """
    excitation_overlap_regions = []
    emission_overlap_regions = []

    # Iterate over all fluorophore pairs
    for f1, f2 in itertools.combinations(fluorophores, 2):
        ex1 = df.loc[f"{f1}", "EX"]
        ex2 = df.loc[f"{f2}", "EX"]
        em1 = df.loc[f"{f1}", "EM"]
        em2 = df.loc[f"{f2}", "EM"]

        ex_overlap = ex1 * ex2
        if ex_overlap.sum() > 0:  # Ensure there's an actual overlap
            excitation_overlap_regions.append((wavelengths, ex_overlap, f"{f1}-{f2}"))

        # Compute emission overlap
        em_overlap = em1 * em2
        if em_overlap.sum() > 0:
            emission_overlap_regions.append((wavelengths, em_overlap, f"{f1}-{f2}"))

    return excitation_overlap_regions, emission_overlap_regions

def compute_crosstalk_regions(wavelengths, df, fluorophores):
    """
    Computes the spectral regions where crosstalk occurs.
    
    Args:
        df (pd.DataFrame): Fluorophore spectral data.
        crosstalk_matrix (pd.DataFrame): Crosstalk values.
        fluorophores (List[str]): Selected fluorophores.

    Returns:
        List[Tuple[List[float], List[float], str]]: Crosstalk region data with x, y, and label.
    """
    crosstalk_regions = []

    for f1, f2 in itertools.product(fluorophores, repeat=2):  # Include all n × n interactions
        if f1 != f2:  # Avoid self-crosstalk
            crosstalk_magnitude = df.loc[f"{f1}","EM"] * df.loc[f"{f2}","EX"]
            if crosstalk_magnitude.sum() > 0:  # Ensure overlap exists
                crosstalk_regions.append((wavelengths, crosstalk_magnitude, f"{f1} → {f2}"))

    return crosstalk_regions

def plot_spectral_overlap_interactive(wavelengths,df, fluorophores):
    """
    Plots fluorophore excitation & emission spectra with highlighted overlap regions using Plotly.
    
    Args:
        df (pd.DataFrame): Spectral data.
        fluorophores (List[str]): Selected fluorophores.
    """
    excitation_overlap_regions, emission_overlap_regions = compute_spectral_overlap_regions(wavelengths, df, fluorophores)
    crosstalk_regions = compute_crosstalk_regions(wavelengths, df, fluorophores)

    fluorophore_colors,_ = get_fluorophore_colors(wavelengths, df, fluorophores)
    fig = go.Figure()

    # Plot individual spectra with assigned colors
    for fluor in fluorophores:
        color = fluorophore_colors[fluor]  # Get fluorophore color
        em = df.loc[f"{fluor}", "EM"]
        ex = df.loc[f"{fluor}", "EX"]

        fig.add_trace(go.Scatter(
            x=wavelengths, y=em, mode='lines',
            name=f"{fluor} Emission", line=dict(width=2, color=color)
        ))
        
        fig.add_trace(go.Scatter(
            x=wavelengths, y=ex, mode='lines',
            name=f"{fluor} Excitation", line=dict(width=2, dash='dash', color=color)
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
    
def get_fluorophore_colors(wavelengths, df, fluorophores):
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

    for fluor in fluorophores:
        em = df.loc[f"{fluor}", "EM"]
        ex = df.loc[f"{fluor}", "EX"]
        
        peak_wavelength = wavelengths[np.argmax(em)]  # Find peak emission wavelength
        fluorophore_colors[fluor] = wavelength_to_rgb(peak_wavelength)
        
        peak_excitation_wavelength = wavelengths[np.argmax(ex)]  # Find peak emission wavelength
        peak_excitation_wavelengths[fluor] = peak_excitation_wavelength

    return fluorophore_colors, peak_excitation_wavelengths
    
def plot_simulated_emission_interactive(wavelengths: list, df: pd.DataFrame, fluorophores: list, lights: list, crosstalk_matrix: np.ndarray):
    """
    Simulates and plots expected emission spectra when exciting fluorophores with given light sources using Plotly.

    Args:
        df (pd.DataFrame): Spectral data with "Excitation" and "Emission" spectra.
        fluorophores (list): Selected fluorophores to simulate.
        lights (list): List of (center_wavelength, bandwidth) for each light source.
    """
    fig = go.Figure()

    i = 0
    for light in lights:
        peak_wavelength = wavelengths[light.argmax()]
        excitation_color = wavelength_to_rgb(peak_wavelength)
        
        # Add excitation light as a dashed line
        fig.add_trace(go.Scatter(
            x=wavelengths, y=light, mode='lines',
            name=f"ExcitationLight:{peak_wavelength}nm)", line=dict(width=2, dash='dash', color=excitation_color)
        ))
        
        for f in fluorophores:
            # expected_emission = compute_expected_emission_crosstalk(df, best_fluorofores, light)
            expected_emission = compute_expected_emission_crosstalk(wavelengths, df, best_fluorofores, light, crosstalk_matrix)
            light = light + expected_emission # not ideal solution, may need to loop over multiple times?
            
        # Add expected emission as solid line
        fig.add_trace(go.Scatter(
            x=wavelengths, y=expected_emission, mode='lines',
            name=f"Total Emission (ExcitationLight:{peak_wavelength}nm)", line=dict(width=2, color=excitation_color)
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
    
def plot_filter_spectra(wavelengths: list, df, fluorophores: list, ex_filters: list, em_filters: list, lights: list):
    """
    Plots fluorophore excitation & emission spectra with highlighted overlap regions using Plotly.
    
    Args:
        df (pd.DataFrame): Spectral data.
        fluorophores (List[str]): Selected fluorophores.
        ex_filters (List[ndarray]): List of excitation filter transmittance spectra
        em_filters (List[ndarray]): List of emission filter transmittance spectra
        lights (List[ndarray]): List of illumination light spectra
    """
    excitation_overlap_regions, emission_overlap_regions = compute_spectral_overlap_regions(wavelengths, df, fluorophores)
    crosstalk_regions = compute_crosstalk_regions(wavelengths, df, fluorophores)

    fluorophore_colors,_ = get_fluorophore_colors(wavelengths, df, fluorophores)
    fig = go.Figure()
    
    # Plot Excitation filter spectra
    if len(ex_filters) == 1:
        ex_filter = ex_filters[0]
        fig.add_trace(go.Scatter(
                x=ex_filter[:,0], y=ex_filter[:,1], mode='lines',
                name=f"Excitation Filter", line=dict(width=2, dash='dash', color="white")
            ))
    else:
        if len(ex_filters) != len(fluorophores):
            raise Exception("Number of filters must be 1 or equal to number of fluorophores")
        
        for fluorophore, ex_filter in zip(fluorophores,ex_filters):
            fig.add_trace(go.Scatter(
                    x=ex_filter[:,0], y=ex_filter[:,1], mode='lines',
                    name=f"{fluorophore} Excitation Filter", line=dict(width=2, dash='dash', color="white")
                ))
            
    # Plot Emission filter spectra
    if len(em_filters) == 1:
        em_filter = em_filters[0]
        fig.add_trace(go.Scatter(
                x=em_filter[:,0], y=em_filter[:,1], mode='lines',
                name=f"Emission Filter", line=dict(width=2, dash='dash', color="gray")
            ))
    else:
        if len(em_filters) != len(fluorophores):
            raise Exception("Number of filters must be 1 or equal to number of fluorophores")
        
        for fluorophore, em_filter in zip(fluorophores,em_filters):
            fig.add_trace(go.Scatter(
                    x=em_filter[:,0], y=em_filter[:,1], mode='lines',
                    name=f"{fluorophore} Emission Filter", line=dict(width=2, dash='dash', color="gray")
                ))

    # Plot individual spectra with assigned colors
    for fluor in fluorophores:
        color = fluorophore_colors[fluor]  # Get fluorophore color
        em, ex = df.loc[f"{fluor}", "EM"], df.loc[f"{fluor}", "EX"]

        fig.add_trace(go.Scatter(
            x=wavelengths, y=em, mode='lines',
            name=f"{fluor} Emission", line=dict(width=2, color=color)
        ))
        
        fig.add_trace(go.Scatter(
            x=wavelengths, y=ex, mode='lines',
            name=f"{fluor} Excitation", line=dict(width=2, dash='dash', color=color)
        ))
            
    for light in lights:
        peak_wavelength = wavelengths[light.argmax()]
        color = wavelength_to_rgb(peak_wavelength)

        fig.add_trace(go.Scatter(
            x=wavelengths, y=light, mode='lines',
            name=f"{peak_wavelength}nm Light", line=dict(width=2, dash='dot', color=color)
        ))

    # Update layout with overlap percentage info
    fig.update_layout(
        title=f"Filter Efficiency Plot",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Intensity/Transmittance",
        template="plotly_dark",
        hovermode="x unified"
    )

    fig.show()
    
def generate_light_spectra(wavelengths, center, width, type="gauss"):
    if type == "band":
        return band_excitation(wavelengths,center,width)
    elif type == "gauss":
        return gaussian_excitation(wavelengths, center, width)
    else:
        raise Exception("Invalid spectral distribution type")

if __name__ == "__main__":
    # Specify fluorophore data files
    fluor_spectra_file = "Fluorophores\\Gentec_All.csv"
    fluor_props_file = "Fluorophores\\Fluorophore_Properties.xlsx"
    
    # Read in fluorophore data
    fluor_spectra = pd.read_csv(fluor_spectra_file)
    fluor_spectra.fillna(0, inplace=True)  # Replace NaNs with 0
    wavelengths = fluor_spectra.iloc[:,0].values
    
    fluor_properties = pd.read_excel(fluor_props_file,index_col=0,usecols="A:C")
    
    fluor_data_list = []
    for fluor in fluor_properties.index:
        try:
            fluor_row = [fluor_properties.loc[fluor,"QE"], fluor_properties.loc[fluor,"ExtCoeff"], fluor_spectra[f"{fluor} EX"].to_numpy(), fluor_spectra[f"{fluor} EM"].to_numpy()]
            fluor_data_list.append(fluor_row)
        except KeyError:
            fluor_row = None
        
    fluor_df = pd.DataFrame(fluor_data_list,index = fluor_properties.index, columns=["QE","ExtCoeff","EX","EM"])

    # Fluorophore Selection Specification
    fluorophore_optimization = True # Select fluorophores to minimize crosstalk and spectral overlap or use manual selection

    num_fluorophores = 3 # Number of fluorophores to use
    exclusion_list = ["ATTO565"]

    crosstalk_matrix = compute_crosstalk(wavelengths,fluor_df)
    crosstalk_matrix = crosstalk_matrix.drop(index=exclusion_list, columns=exclusion_list)
    
    # pareto_solutions = find_optimal_fluorophore_set_MOO(crosstalk_matrix, # TODO: verify pareto optimization
    #                              df, 
    #                              num_fluorophores, 
    #                              population_size=10) 
    
    # print(pareto_solutions)
    
    
    if fluorophore_optimization: # Optimize selection
        # Select best fluorophores
        best_fluorofores, best_fluorofores_matrix, min_crosstalk, min_overlap = find_optimal_fluorophore_set(wavelengths, fluor_df, crosstalk_matrix, num_fluorophores)

    else: # manual selection
        best_fluorofores = ["FAM","ROX","Cy5"]
        
        best_fluorofores_matrix = crosstalk_matrix.loc[list(best_fluorofores), list(best_fluorofores)]
        
    print("\nOptimal fluorophore combination with minimum crosstalk:")
    print(best_fluorofores)
    print("\nCrosstalk matrix:")
    print(best_fluorofores_matrix)

    # Plot spectra of selected fluorophores
    plot_spectral_overlap_interactive(wavelengths,fluor_df, best_fluorofores)
    
    # Illumination specification
    _, peak_excitation_wavelengths = get_fluorophore_colors(wavelengths,fluor_df, best_fluorofores)
    
    # LED_data_files = ["LEDS\\XEG-CYAN_Processed.csv", "LEDS\\XEG-AMBER_Processed.csv", "LEDS\\XEG-PHOTORED_Processed.csv"]
    # lights = [np.genfromtxt(file_name,delimiter=",",skip_header=True)[:,1] for file_name in LED_data_files] # Imported spectra
    lights = [generate_light_spectra(wavelengths, center, 10, type="band") for center in peak_excitation_wavelengths.values()] # Generated spectra
    
    # Excitation and emission simulation
    excitation_efficiency = compute_excitation_efficiency(wavelengths, fluor_df, best_fluorofores, lights)  # Compute excitation efficiency using lights directly
    print("\nExcitation Efficiency")
    print(tabulate(list(zip(best_fluorofores,excitation_efficiency)),headers=["Fluorophore","Excitation Efficiency"]))
    
    plot_simulated_emission_interactive(wavelengths, fluor_df, best_fluorofores, lights, best_fluorofores_matrix)
    
    # Filter Evaluation
    ex_filters = split_multiband_filter(np.genfromtxt("Filters\\TriBandpassFAMROXCy5EX.csv",delimiter=',',skip_header=True))
    em_filters = split_multiband_filter(np.genfromtxt("Filters\\TriBandpassFAMROXCy5EM.csv",delimiter=',',skip_header=True))
    
    plot_filter_spectra(wavelengths, fluor_df, best_fluorofores, ex_filters, em_filters, lights)
    
    # Compute LED-excitation filter efficiency (% of LED light that is transmitted through filter) 
    if len(ex_filters) == 1:
        LED_EX_Filter_efficiency = [normalized_product_integral(wavelengths,light,ex_filters[0][:,1]) for light in lights]
    else:
        LED_EX_Filter_efficiency = [normalized_product_integral(wavelengths,light,ex_filter[:,1]) for light,ex_filter in zip(lights,ex_filters)]
    
    print("\nLED-EX Filter Efficiency")
    print(tabulate(np.transpose(np.expand_dims(LED_EX_Filter_efficiency,axis=1)),headers=best_fluorofores))
    
    # Compute Excitation Filter - fluorophore excitation overlap (% of fluorophore excitation spectrum excited by light transmitted through EX filter) - determines if EX filter has correct band
    if len(ex_filters) == 1:
        EX_Filter_efficiency = [normalized_product_integral(wavelengths,normalizeArray(ex_filters[0][:,1]),fluor_df.loc[f"{fluor}","EX"]) for fluor in best_fluorofores] # TODO: normalizing multiband filter should be specific to the band being integrated
    else:
        EX_Filter_efficiency = [normalized_product_integral(wavelengths,normalizeArray(ex_filter[:,1]),fluor_df.loc[f"{fluor}","EX"]) for fluor,ex_filter  in zip(best_fluorofores,ex_filters)]
    
    print("\nEX Filter - Fluorophore EX Efficiency")
    print(tabulate(np.transpose(np.expand_dims(EX_Filter_efficiency,axis=1)),headers=best_fluorofores))
    
    # Compute fluorophore emission - emission filter overlap (% of emitted spectrum that is transmitted through excitation filter) - determines if EM filter has correct band
    if len(em_filters) == 1:
        EM_Filter_efficiency = [normalized_product_integral(wavelengths,fluor_df.loc[f"{fluor}","EM"],normalizeArray(em_filters[0][:,1])) for fluor in best_fluorofores]
    else:
        EM_Filter_efficiency = [normalized_product_integral(wavelengths,fluor_df.loc[f"{fluor}","EX"],normalizeArray(em_filter[:,1])) for fluor,em_filter  in zip(best_fluorofores,em_filters)]
    
    print("\nFluorophore EM - EM Filter Efficiency")
    print(tabulate(np.transpose(np.expand_dims(EM_Filter_efficiency,axis=1)),headers=best_fluorofores))
    
    # Compute excitation-emission filter crosstalk (% of excitation light passing through emission filter) - should be 0
    combination = []
    EX_EM_overlap = []
    for f1, f2 in itertools.product([0,1,2], [0,1,2]):
        combination.append(f"{best_fluorofores[f1]} EX - {best_fluorofores[f2]} EM")
        EX_EM_overlap.append(normalized_product_integral(wavelengths,normalizeArray(ex_filters[f1][:,1]),normalizeArray(em_filters[f2][:,1])))
        
    print("EX Filter - EM Filter overlap")
    print(tabulate(list(zip(combination, EX_EM_overlap)),headers=["Combination","Overlap"]))
    
    