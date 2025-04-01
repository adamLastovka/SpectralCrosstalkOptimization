from main import *
import numpy as np
import pandas as pd
import itertools
import plotly.graph_objects as go
import os
from dataclasses import dataclass
from tqdm import tqdm
from alive_progress import alive_bar
import pickle
from datetime import datetime

@dataclass
class Component():
    """Stores metadata about optical components"""
    index: int
    spectrum: np.ndarray
    spectral_range: Tuple[float, float]
    peak_wavelength: float
    
    def __init__(self, index, wavelengths, spectrum, type):
        self.index = index
        self.spectrum = spectrum
        self.type = type
        
        self.spectral_range = self.calculate_spectral_range(wavelengths, spectrum, 0.001)  # (start_wl, end_wl) at 0.1% threshold
        
        if self.type == "band":
            self.peak_wavelength = np.mean(self.spectral_range)
        else:
            self.peak_wavelength = wavelengths[np.argmax(spectrum)]
    
    def calculate_spectral_range(self, wavelengths: np.ndarray, spectrum: np.ndarray, threshold=0.01) -> tuple:
        """Calculate spectral range where intensity > threshold% of max"""
        threshold_val = np.max(spectrum) * threshold
        mask = spectrum >= threshold_val
        if not np.any(mask):
            return (0.0, 0.0)
        
        indices = np.where(mask)[0]
        return (wavelengths[indices[0]], wavelengths[indices[-1]])

def calculate_emission_power(wavelengths, df, fluorophores, C_vect, lights):
    """
    Calculates emission power when exciting fluorophores with given light sources.
    Extracted from plot_emission_power function without plotting.
    
    Returns:
        float: Total emission power
        list: Power per fluorophore
    """
    mean_em_power = 0
    coupled_ex_power = []
    em_power_per_fluorophore = []
    
    for i, (light, C) in enumerate(zip(lights, C_vect)):
        E = np.vstack([normalize_area(wavelengths, df.loc[f"{fluor}", "EX"]) for fluor in fluorophores])
        M = np.vstack([normalize_area(wavelengths, df.loc[f"{fluor}", "EM"]) for fluor in fluorophores])
        phi = np.array([df.loc[f"{fluor}", "QE"] for fluor in fluorophores])
        epsilon = np.array([df.loc[f"{fluor}", "ExtCoeff"] for fluor in fluorophores])
        
        S = np.outer(epsilon*C, epsilon*C)
        
        I_ex_initial = E @ light
        
        A = epsilon * C * WELL_DEPTH
        fraction_absorbed = 1 - 10**-A
        I_ex = I_ex_initial * fraction_absorbed
        
        X = M @ E.T
        scaled_X = X * S
        F = np.diag(phi) @ scaled_X
        D = np.diag(phi) @ I_ex
        O = np.linalg.solve(np.eye(len(fluorophores)) - F, D)
        
        # final_spectrum = M.T @ O
        # fluor_power = np.trapezoid(final_spectrum, wavelengths)

        coupled_ex_power.append(I_ex)
        em_power_per_fluorophore.append(O)
        mean_em_power += sum(O)
    
    return coupled_ex_power, mean_em_power, em_power_per_fluorophore

def generate_combinations(filtered_components: dict, fluorophores: list):
    """
    Generates combinations where each element is chosen from the filtered components
    specific to its corresponding fluorophore.
    
    Args:
        filtered_components: Dictionary from filter_components_by_overlap
        fluorophores: List of fluorophores in order
    
    Returns:
        configs [List[Dict]]: List of all viable configurations
    """
    # Get component lists for each fluorophore in order
    led_lists = [filtered_components[fluor]['leds'] for fluor in fluorophores]
    ex_filter_lists = [filtered_components[fluor]['ex_filters'] for fluor in fluorophores]
    em_filter_lists = [filtered_components[fluor]['em_filters'] for fluor in fluorophores]
    
    # Generate Cartesian product of component indices
    led_combos = list(itertools.product(*led_lists))
    ex_filter_combos = list(itertools.product(*ex_filter_lists))
    em_filter_combos = list(itertools.product(*em_filter_lists))
    
    print(f"Evaluating {len(led_combos)} LED combinations")
    print(f"Evaluating {len(ex_filter_combos)} excitation filter combinations")
    print(f"Evaluating {len(em_filter_combos)} emission filter combinations")
    
    num_configs = len(led_combos) * len(ex_filter_combos) * len(em_filter_combos)
    print(f"Total configurations to evaluate: {num_configs}")
    
    # Eliminate Combos with filter bleedthrough
    configs = []
    for ex_combo in ex_filter_combos:
        for em_combo in em_filter_combos:
            # Check filter bleedtrhough and construct list of filtered configs
            for ex_filter, em_filter in zip(ex_combo,em_combo):
                if ex_filter.spectral_range[1] >= em_filter.spectral_range[0]:
                    break
            else: # only executes if there is no bleedthrough    
                for led_combo in led_combos: 
                    configs.append({'led_combo':led_combo,'ex_combo':ex_combo,'em_combo':em_combo}) # only append if no bleedthrough
                    
    print(f"Eliminated {num_configs-len(configs)} filter combinations with bleedthrough")
    
    # Convert to list of lists with limit
    return configs

def optimize_filters_and_leds(wavelengths, fluor_df, fluorophores, fluor_conc, filtered_components, led_power=None):
    """
    Optimizes filter and LED combinations to maximize emission power.
    """
    if led_power is None:
        led_power = [1.0] * len(fluorophores)
    
    # Results storage
    results = []
    
    # Generate combinations
    configurations = generate_combinations(filtered_components, fluorophores)
    
    # Iterate through combinations
    with alive_bar(len(configurations)) as bar:
        for config_id,config in enumerate(configurations):
            led_combo = config['led_combo']
            ex_combo = config['ex_combo']
            em_combo = config['em_combo']
            
            # Apply excitation filters to LEDs
            filtered_leds = []
            for i, (ex_filter, led) in enumerate(zip(ex_combo, led_combo)):
                norm_light = normalize_area(wavelengths, led.spectrum) * led_power[i]  # Normalize so that total area = P
                filtered_leds.append(ex_filter.spectrum * norm_light)                    
                
            # Calculate emission power
            coupled_ex_power, mean_em_power, em_power_per_fluorophore = calculate_emission_power(
                wavelengths, fluor_df, fluorophores, fluor_conc, filtered_leds)
            
            # Calculate additional metrics
            metrics = {}
            
            metrics['coupled_ex_power'] = coupled_ex_power 
            
            # LED-EX Filter efficiency
            led_ex_efficiency = [normalized_product_integral(wavelengths, led.spectrum, ex_filter.spectrum) 
                                for led, ex_filter in zip(led_combo, ex_combo)]
            metrics['led_ex_efficiency'] = np.mean(led_ex_efficiency)
            
            # EX Filter - fluorophore excitation overlap
            ex_fluor_efficiency = [normalized_product_integral(wavelengths, normalizeArray(ex_filter.spectrum), 
                                fluor_df.loc[f"{fluor}", "EX"]) 
                                for fluor, ex_filter in zip(fluorophores, ex_combo)]
            metrics['ex_fluor_efficiency'] = np.mean(ex_fluor_efficiency)
            
            # Fluorophore emission - emission filter overlap
            em_filter_efficiency = [normalized_product_integral(wavelengths, fluor_df.loc[f"{fluor}", "EM"], 
                                normalizeArray(em_filter.spectrum)) 
                                for fluor, em_filter in zip(fluorophores, em_combo)]
            metrics['em_filter_efficiency'] = em_filter_efficiency
            
            # Store result
            results.append({
                'config_id': config_id,
                'configuration': config,
                'mean_em_power': mean_em_power,
                'em_power_per_fluorophore': em_power_per_fluorophore,
                **metrics
            })
            
            bar()
            
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def plot_optimization_results(results_df,wavelengths,fluor_df,fluorophores,led_power):
    """
    Creates Pareto plots to visualize optimization results.
    """
    # Create Pareto scatter plot
    fig = go.Figure()
    
    # Add scatter plot for total emission power vs. emission filter efficiency
    for i, fluorophore in enumerate(fluorophores):
        fig.add_trace(go.Scatter(
            x=list(map(lambda x: x[i][i],results_df['em_power_per_fluorophore'][:])),
            y=list(map(lambda x: x[i],results_df['em_filter_efficiency'])),
            mode='markers',
            marker=dict(size=10),
            text=results_df['config_id'],
            hovertemplate="<b>Config ID:</b> %{text}<br>" +
                        "<b>Emission Power:</b> %{x:.2e}<extra></extra><br>" +
                        "<b>Emission Filter Efficiency:</b> %{y:.2f}",  
            name=f"{fluorophore}"
        ))
        
    fig.add_trace(go.Scatter(
        x=results_df['mean_em_power'],
        y=list(map(lambda x: np.mean(x),results_df['em_filter_efficiency'])),
        mode='markers',
        text=results_df['config_id'],
        hovertemplate="<b>Config ID:</b> %{text}<br>" +
                        "<b>Emission Power:</b> %{x:.2e}<extra></extra><br>" +
                        "<b>Emission Filter Efficiency:</b> %{y:.2f}", 
        marker=dict(size=10),
        name=f"Mean"
    ))
    
    # Highlight the best configuration
    best_idx = results_df['mean_em_power'].idxmax()
    fig.add_trace(go.Scatter(
        x=[results_df.loc[best_idx, 'em_power_per_fluorophore']],
        y=[results_df.loc[best_idx, 'em_filter_efficiency']],
        mode='markers',
        marker=dict(size=15, color='red', symbol='star'),
        name='Best Configuration'
    ))
    
    fig.update_layout(
        title='Filter & LED Optimization Results',
        xaxis_title='Emission Power',
        yaxis_title='Emission Filter Efficiency',
        template='plotly_dark',
        hovermode='closest'
    )
    
    fig.show()
    
    # Create parallel coordinates plot
    dimensions = [
        dict(label='Config ID', values=list(range(len(results_df))),  # Add config IDs as a categorical axis
             ticktext=results_df['config_id'], tickvals=list(range(len(results_df)))),
        dict(range=[0, results_df['mean_em_power'].max()],
             label='Total Emission Power', values=results_df['mean_em_power']),
        dict(range=[0, 1],
             label='LED-EX Efficiency', values=results_df['led_ex_efficiency']),
        dict(range=[0, 1],
             label='EX-Fluor Efficiency', values=results_df['ex_fluor_efficiency']),
        dict(range=[0, 1],
             label='EM Filter Efficiency', values=list(map(lambda x: np.mean(x),results_df['em_filter_efficiency'])))
    ]
    
    fig2 = go.Figure(data=go.Parcoords(
        line=dict(color=results_df['mean_em_power'],
                colorscale='Viridis',
                showscale=True),
                dimensions=dimensions,
    ))
    
    fig2.update_layout(
        title='Multi-Dimensional Optimization Results',
        template='plotly_dark'
    )
    
    fig2.show()
    
    # Display the best configuration
    best_config = results_df.loc[best_idx, 'configuration']
    best_ex_filters = best_config["ex_combo"]
    best_em_filters = best_config["em_combo"]
    best_leds = best_config["led_combo"]
    
    print(f"\nBest Configuration:")
    print(f"Total Power: {results_df.loc[best_idx, 'mean_em_power']:.3e}")
    print(f"LED-EX Efficiency: {results_df.loc[best_idx, 'led_ex_efficiency']:.2f}")
    print(f"EX-Fluor Efficiency: {results_df.loc[best_idx, 'ex_fluor_efficiency']:.2f}")
    print(f"EM Filter Efficiency: {results_df.loc[best_idx, 'em_filter_efficiency']:.2f}")
    
    # Plot the best configuration using existing function
    plot_filter_spectra(wavelengths, fluor_df, fluorophores, best_ex_filters, best_em_filters, best_leds)
    
    # Also plot the emission power for the best configuration
    best_filtered_leds = [led*filter for led, filter in zip(best_leds,best_ex_filters)]
    plot_emission_power(wavelengths, fluor_df, fluorophores, fluor_conc, best_filtered_leds, led_power)
    
def filter_components_by_overlap(wavelengths, fluor_df, fluorophores, available_leds, 
                                     available_ex_filters, available_em_filters, threshold=0.1):
    """
    Filter LEDs, excitation filters, and emission filters based on overlap with fluorophores.
    Returns a dictionary keyed by fluorophore names, with lists of indices of viable filters and LEDs.
    
    Args:
        wavelengths (np.ndarray): Array of wavelengths
        fluor_df (pd.DataFrame): DataFrame containing fluorophore spectral data
        fluorophores (list): List of fluorophore names to consider
        available_leds (list): List of LED spectra
        available_ex_filters (list): List of excitation filter spectra
        available_em_filters (list): List of emission filter spectra
        threshold (float): Minimum overlap percentage required (0.1 = 10%)
        
    Returns:
        dict: Dictionary keyed by fluorophore name, each containing lists of indices
              for compatible LEDs, excitation filters, and emission filters
    """
    filtered_components = {fluor: {'leds': [], 'ex_filters': [], 'em_filters': []} for fluor in fluorophores}

    
    for fluor in fluorophores:
        # Filter LEDs that have >threshold overlap with each fluorophore's excitation spectrum
        for i, led in enumerate(available_leds):
            if normalized_product_integral(wavelengths, led.spectrum, fluor_df.loc[f"{fluor}", "EX"]) > threshold:
                filtered_components[fluor]['leds'].append(led)

        # Filter excitation filters with >threshold overlap with each fluorophore's excitation spectrum
        for i, ex_filter in enumerate(available_ex_filters):
            if normalized_product_integral(wavelengths, ex_filter.spectrum, fluor_df.loc[f"{fluor}", "EX"]) > threshold:
                filtered_components[fluor]['ex_filters'].append(ex_filter)

        # Filter emission filters with >threshold overlap with each fluorophore's emission spectrum
        for i, em_filter in enumerate(available_em_filters):
                if normalized_product_integral(wavelengths, fluor_df.loc[f"{fluor}", "EM"], em_filter.spectrum) > threshold:
                    filtered_components[fluor]['em_filters'].append(em_filter)

    # Print summary statistics
        print(f"{fluor}: {len(filtered_components[fluor]['leds'])}/{len(available_leds)} LEDs, "
              f"{len(filtered_components[fluor]['ex_filters'])}/{len(available_ex_filters)} excitation filters, "
              f"{len(filtered_components[fluor]['em_filters'])}/{len(available_em_filters)} emission filters")
    
    return filtered_components
    
### Filter and LED Optimization ###
if __name__ == "__main__":
    run_sim = True # Run sim or load results
    load_file = 'OptimResults\\simulation_results_1.pkl'
    
    if run_sim:
        # Specify fluorophore data files
        fluor_spectra_file = "Fluorophores\\Gentec_All.csv"
        fluor_props_file = "Fluorophores\\Fluorophore_Properties.xlsx"
        
        wavelengths, fluor_df = generate_fluor_df(fluor_spectra_file, fluor_props_file)
        
        # Select fluorophores
        fluor_selection = ["FAM", "TexasRed", "Cy5_5"]
        
        num_fluorophores = len(fluor_selection)
            
        fluor_conc = [100e-9] * num_fluorophores
        
        # Define candidate LEDs for optimization
        led_power = [1] * num_fluorophores
        
        available_leds_files = [os.path.join("LEDS", f) for f in os.listdir("LEDS") if os.path.isfile(os.path.join("LEDS", f))]
        available_leds = []
        
        # Add spectra from files
        for i,file_name in enumerate(available_leds_files):
            spectrum = np.genfromtxt(file_name, delimiter=",", skip_header=True)[:,1]
            available_leds.append(Component(i, wavelengths, spectrum, type="import"))
            
        # Define candidate excitation filters
        available_ex_filters = []
        
        # Add existing filters
        multiband_filters = split_multiband_filter(wavelengths,np.genfromtxt("Filters\\TriBandpassFAMROXCy5EX.csv",delimiter=',',skip_header=True)[:,1])
        for i,filter in enumerate(multiband_filters):
            available_ex_filters.append(Component(i, wavelengths, filter, type="import"))
        
        num_filters = len(available_ex_filters)
        for i,center in enumerate([455, 495, 535, 550, 575, 590, 600, 655,680]):
            spectrum = generate_light_spectra(wavelengths, center, 10, type="band")
            available_ex_filters.append(Component(i+num_filters, wavelengths, spectrum, type="band"))
        
        # Add generated filters
        num_filters = len(available_ex_filters)
        for i,center in enumerate([470, 480, 580, 590, 670, 680]):
            spectrum = generate_light_spectra(wavelengths, center, 20, type="band")
            available_ex_filters.append(Component(i+num_filters, wavelengths, spectrum, type="band"))
        
        # Define candidate emission filters
        available_em_filters = []
        
        # Add existing filters
        for center in [455, 495, 535, 550, 575, 590, 600, 655,680]:
            spectrum = generate_light_spectra(wavelengths, center, 10, type="band")
            available_em_filters.append(Component(i, wavelengths, spectrum,type="band"))
        
        # Add generated filters
        num_filters = len(available_em_filters)
        for i,center in enumerate([520,530,620,630,700,720]):
            spectrum = generate_light_spectra(wavelengths, center, 25, type="band")
            available_em_filters.append(Component(i+num_filters, wavelengths, spectrum, type="band"))
        
        ### Pre filtering ###
        overlap_threshold = 0.2
        filtered_components = filter_components_by_overlap(wavelengths, fluor_df, fluor_selection, available_leds, 
                                        available_ex_filters, available_em_filters, overlap_threshold)
        
        # Run optimization
        print("Running filter and LED optimization...")
        optimization_results = optimize_filters_and_leds(
            wavelengths,
            fluor_df,
            fluor_selection,
            fluor_conc,
            filtered_components,
            led_power
        )
        
        # Save the results to a pickle file
        save_file = f"OptimResults\\Results{fluor_selection[0]}{fluor_selection[1]}{fluor_selection[2]}_{datetime.now().strftime('%m-%d_%H-%M-%S')}"
        
        result_data = [optimization_results,wavelengths,wavelengths,fluor_df,fluor_selection,led_power]
        with open(save_file, 'wb') as f:
            pickle.dump(result_data, f)

    else:
        with open(load_file, 'rb') as f:
            optimization_results,wavelengths,wavelengths,fluor_df,fluor_selection,led_power = pickle.load(f)

    # Plot Pareto front
    plot_optimization_results(optimization_results,wavelengths,fluor_df,fluor_selection,led_power)
    
    # Find and use the best configuration
    best_idx = optimization_results['total_power'].idxmax()
    optimal_ex_filters = optimization_results.loc[best_idx, 'ex_filters']
    optimal_em_filters = optimization_results.loc[best_idx, 'em_filters']
    optimal_leds = optimization_results.loc[best_idx, 'leds']
    
    print("\nUsing optimal LED and filter configuration for final analysis")
