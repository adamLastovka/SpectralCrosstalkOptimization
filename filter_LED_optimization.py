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
    name: str
    
    def __init__(self, index, wavelengths, spectrum, type, name=None):
        self.index = index
        self.spectrum = spectrum
        self.type = type
        self.name = name
        
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
    total_em_power = 0
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
    
    return coupled_ex_power, em_power_per_fluorophore

def plot_filter_spectra_components(wavelengths: list, df, fluorophores: list, ex_filters: list, em_filters: list, lights: list):
    """
    Plots fluorophore excitation & emission spectra with highlighted overlap regions using Plotly.
    
    Args:
        df (pd.DataFrame): Spectral data.
        fluorophores (List[str]): Selected fluorophores.
        ex_filters (List[Component]): List of excitation filter component objects
        em_filters (List[Component]): List of emission filter component objects
        lights (List[Component]): List of illumination light component objects
    """
    fluorophore_colors,_,_ = get_fluorophore_colors(wavelengths, df, fluorophores)
    fig = go.Figure()
    
    # Plot Excitation filter spectra
    if len(ex_filters) == 1:
        ex_filter = ex_filters[0]
        fig.add_trace(go.Scatter(
                x=wavelengths, y=ex_filter.spectrum, mode='lines',
                name=f"{ex_filter.name}", line=dict(width=2, dash='dash', color="white")
            ))
    else:
        if len(ex_filters) != len(fluorophores):
            raise Exception("Number of filters must be 1 or equal to number of fluorophores")
        
        for fluorophore, ex_filter in zip(fluorophores,ex_filters):
            fig.add_trace(go.Scatter(
                    x=wavelengths, y=ex_filter.spectrum, mode='lines',
                    name=f"{ex_filter.name}", line=dict(width=2, dash='dash', color="white")
                ))
            
    # Plot Emission filter spectra
    if len(em_filters) == 1:
        em_filter = em_filters[0]
        fig.add_trace(go.Scatter(
                x=wavelengths, y=em_filter.spectrum, mode='lines',
                name=f"{em_filter.name}", line=dict(width=2, dash='dash', color="gray")
            ))
    else:
        if len(em_filters) != len(fluorophores):
            raise Exception("Number of filters must be 1 or equal to number of fluorophores")
        
        for fluorophore, em_filter in zip(fluorophores,em_filters):
            fig.add_trace(go.Scatter(
                    x=wavelengths, y=em_filter.spectrum, mode='lines',
                    name=f"{em_filter.name}", line=dict(width=2, dash='dash', color="gray")
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
        peak_wavelength = wavelengths[light.spectrum.argmax()]
        color = wavelength_to_rgb(peak_wavelength)

        fig.add_trace(go.Scatter(
            x=wavelengths, y=light.spectrum, mode='lines',
            name=f"{light.name}", line=dict(width=2, dash='dot', color=color)
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
    
def plot_emission_power_components(wavelengths: list, df: pd.DataFrame, fluorophores: list, C_vect: list, ex_filters: list, lights: list, P_vect: float):
    """
    Simulates and plots expected emission spectra when exciting fluorophores with given light sources using Plotly.

    Args:
        df (pd.DataFrame): Spectral data with "Excitation" and "Emission" spectra.
        fluorophores (list): Selected fluorophores to simulate.
        C_vect (list): List of fluorophore concentrations
        lights (List[Component]): List of excitation light component objects
        ex_filters (List[Component]): List of excitation filter component objects
        power: List of total emission power of each LED
    """
    fig = go.Figure()
    
    for i, (light,ex_filter,P,C) in enumerate(zip(lights,ex_filters,P_vect,C_vect)):
        filtered_light = ex_filter.spectrum * light.spectrum
        
        norm_light = normalize_area(wavelengths,filtered_light)*P # Normalize so that total area = P
        
        E = np.vstack([normalize_area(wavelengths,df.loc[f"{fluor}", "EX"]) for fluor in fluorophores])  # Excitation Spectra [nFluor × nWavelengths]
        M = np.vstack([normalize_area(wavelengths,df.loc[f"{fluor}", "EM"]) for fluor in fluorophores])  # Emission Spectra [nFluor × nWavelengths]
        phi = np.array([df.loc[f"{fluor}", "QE"] for fluor in fluorophores]) # Fluorophore efficiency [nFluor x 1]
        epsilon = np.array([df.loc[f"{fluor}", "ExtCoeff"] for fluor in fluorophores]) 
        
        S = np.outer(epsilon*C, epsilon*C) # Concentration dependent excitation scaling matrix
        
        I_ex_initial = E @ norm_light # Incident light flux spectrally coupled into each fluorofore [W] [nFluor x 1]
        
        A = epsilon * C * WELL_DEPTH # Absorbance
        fraction_absorbed = 1 - 10**-A # Beer-lambert law
        I_ex = I_ex_initial * fraction_absorbed   # Incident light absorbed by fluorophore [nFluor × 1]

        X = M @ E.T  # [nFluor x nFluor] crosstalk matrix
        scaled_X = X * S # Crosstalk accouting for concentrations and extinction coeff
        F = np.diag(phi) @ scaled_X  # Feedback operator [nFluor x n Fluor] 
        D = np.diag(phi) @ I_ex # Direct excitation (no crosstalk) [nFluor x 1]
        O = np.linalg.solve(np.eye(len(fluorophores)) - F, D)  # Closed form solution for infinite feedback series [nFluor x 1]
        
        # n = 2  # 
        # total_emission = sum(np.linalg.matrix_power(F, k) @ D for k in range(n+1)) # nth order crosstalk emission 
                        
        final_spectrum = M.T @ O  # [nWavelengths × 1] emission spectrum
        
        emission_color = wavelength_to_rgb(wavelengths[final_spectrum.argmax()])
        excitation_color = wavelength_to_rgb(wavelengths[light.spectrum.argmax()])
        
        fig.add_trace(go.Scatter(
            x=wavelengths, y=norm_light, mode='lines',
            name=f" Filtered {light.name} Intensity Distribution [W]", line=dict(width=2, color=excitation_color)
        ))
        
        fig.add_trace(go.Scatter(
            x=wavelengths, y=final_spectrum, mode='lines',
            name=f"Emission response of {light.name} [W]", line=dict(width=2, color=emission_color)
        ))

    fig.update_layout(
        title="Simulated Emission Spectrum Under Multi-Source Excitation",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Intensity",
        template="plotly_dark",
        hovermode="x unified"
    )

    fig.show()

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
            coupled_ex_power, em_power_per_fluorophore = calculate_emission_power(
                wavelengths, fluor_df, fluorophores, fluor_conc, filtered_leds)
            
            mean_em_power = np.mean(np.sum(em_power_per_fluorophore, axis = 0))
            
            # Calculate additional metrics
            metrics = {}
            
            metrics['coupled_ex_power'] = coupled_ex_power 
            
            # LED-EX Filter efficiency
            led_ex_efficiency = [normalized_product_integral(wavelengths, led.spectrum, ex_filter.spectrum) 
                                for led, ex_filter in zip(led_combo, ex_combo)]
            metrics['led_ex_efficiency'] = led_ex_efficiency
            metrics['mean_led_ex_efficiency'] = np.mean(led_ex_efficiency)
            
            # EX Filter - fluorophore excitation overlap
            ex_fluor_efficiency = [normalized_product_integral(wavelengths, normalizeArray(ex_filter.spectrum), 
                                fluor_df.loc[f"{fluor}", "EX"]) 
                                for fluor, ex_filter in zip(fluorophores, ex_combo)]
            metrics['ex_fluor_efficiency'] = ex_fluor_efficiency
            metrics['mean_ex_fluor_efficiency'] = np.mean(ex_fluor_efficiency)
            
            # Fluorophore emission - emission filter overlap
            em_filter_efficiency = [normalized_product_integral(wavelengths, fluor_df.loc[f"{fluor}", "EM"], 
                                normalizeArray(em_filter.spectrum)) 
                                for fluor, em_filter in zip(fluorophores, em_combo)]
            metrics['em_filter_efficiency'] = em_filter_efficiency
            metrics['mean_em_filter_efficiency'] = np.mean(em_filter_efficiency)
            
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

def plot_optimization_results(results_df,wavelengths,fluor_df,fluorophores,fluor_conc,led_power, hide_plots=False):
    """
    Creates Pareto plots to visualize optimization results.
    """
    # Create Pareto scatter plot
    fig = go.Figure()
    
    max_power_idx = results_df[results_df['mean_em_power'] == results_df['mean_em_power'].max()].index
    best_idx = results_df.loc[max_power_idx,'mean_em_filter_efficiency'].idxmax()
    
    if not hide_plots:
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
            
            # Highlight the best result
            fig.add_trace(go.Scatter(
                x=[results_df.loc[best_idx, 'em_power_per_fluorophore'][i][i]],
                y=[results_df.loc[best_idx, 'em_filter_efficiency'][i]],
                mode='markers',
                marker=dict(size=15, color='white', symbol='star'),
                name='Best Configuration'
            ))
            
        fig.add_trace(go.Scatter(
            x=results_df['mean_em_power'],
            y=results_df["mean_em_filter_efficiency"],
            mode='markers',
            text=results_df['config_id'],
            hovertemplate="<b>Config ID:</b> %{text}<br>" +
                            "<b>Emission Power:</b> %{x:.2e}<extra></extra><br>" +
                            "<b>Emission Filter Efficiency:</b> %{y:.2f}", 
            marker=dict(size=10),
            name=f"Mean"
        ))
        
        # Highlight the best result
        fig.add_trace(go.Scatter(
            x=[results_df.loc[best_idx, 'mean_em_power']],
            y=[results_df.loc[best_idx, 'mean_em_filter_efficiency']],
            mode='markers',
            marker=dict(size=15, color='white', symbol='star'),
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
            dict(range=[0, results_df['mean_em_power'].max()],
                label='Total Emission Power', values=results_df['mean_em_power']),
            dict(range=[0, 1],
                label='LED-EX Efficiency', values=results_df['mean_led_ex_efficiency']),
            dict(range=[0, 1],
                label='EX-Fluor Efficiency', values=results_df['mean_ex_fluor_efficiency']),
            dict(range=[0, 1],
                label='EM Filter Efficiency', values=results_df['mean_em_filter_efficiency'])
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
    
    # Display metrics for best config
    print(f"\nBest Configuration:")
    print(f"Config ID: {results_df.loc[best_idx, 'config_id']}")
    
    table_headers = ["Fluorophore", "LEDs", "LED-EX Efficiency", "EX Filters", "EX filter-Fluor EX Efficiency", "EM filters", "Fluor EM-filter Efficiency"]
    led_names = list(map(lambda x: x.name, best_leds))
    led_ex_eff = results_df.loc[best_idx, 'led_ex_efficiency']
    ex_filter_names = list(map(lambda x: x.name, best_ex_filters))
    ex_filter_eff = results_df.loc[best_idx, 'ex_fluor_efficiency']
    em_filter_names = list(map(lambda x: x.name, best_em_filters))
    em_filter_eff = results_df.loc[best_idx, 'em_filter_efficiency']
    
    print(tabulate(list(zip(fluorophores,led_names,led_ex_eff,ex_filter_names,ex_filter_eff,em_filter_names,em_filter_eff)),headers=table_headers))
    
    table_headers = ["Fluorophore","Power_per_fluor","detectable_power","EX Filters", "EM filters"]
    results_df.loc[best_idx, 'em_power_per_fluorophore']
    detectable_power = [f"{np.dot(p,eff):.2e}" for p, eff in zip(results_df.loc[best_idx, 'em_power_per_fluorophore'], [results_df.loc[best_idx, 'em_filter_efficiency']]*3)]
    
    print(tabulate(list(zip(fluorophores,list(map(str,results_df.loc[best_idx, 'em_power_per_fluorophore'])),detectable_power)),headers=table_headers))

    print(f"Mean EM Power: {results_df.loc[best_idx, 'mean_em_power']:.3e}")
    print(f"Mean LED-EX Efficiency: {results_df.loc[best_idx, 'mean_led_ex_efficiency']:.2f}")
    print(f"Mean EX Filter-Fluor Efficiency: {results_df.loc[best_idx, 'mean_ex_fluor_efficiency']:.2f}")
    print(f"Mean fluor-EM Filter Efficiency: {results_df.loc[best_idx, 'mean_em_filter_efficiency']:.2f}")
    
    # Plot the best configuration using existing function
    plot_filter_spectra_components(wavelengths, fluor_df, fluorophores, best_ex_filters, best_em_filters, best_leds)
    
    # Also plot the emission power for the best configuration
    plot_emission_power_components(wavelengths, fluor_df, fluorophores, fluor_conc, best_ex_filters, best_leds, led_power)
    
def filter_components_by_overlap(wavelengths, fluor_df, fluorophores, available_leds, 
                                     available_ex_filters, available_em_filters, LED_threshold, ex_threshold, em_threshold):
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
            if normalized_product_integral(wavelengths, led.spectrum, fluor_df.loc[f"{fluor}", "EX"]) > LED_threshold:
                filtered_components[fluor]['leds'].append(led)

        # Filter excitation filters with >threshold overlap with each fluorophore's excitation spectrum
        for i, ex_filter in enumerate(available_ex_filters):
            if normalized_product_integral(wavelengths, ex_filter.spectrum, fluor_df.loc[f"{fluor}", "EX"]) > ex_threshold:
                filtered_components[fluor]['ex_filters'].append(ex_filter)

        # Filter emission filters with >threshold overlap with each fluorophore's emission spectrum
        for i, em_filter in enumerate(available_em_filters):
                if normalized_product_integral(wavelengths, fluor_df.loc[f"{fluor}", "EM"], em_filter.spectrum) > em_threshold:
                    filtered_components[fluor]['em_filters'].append(em_filter)

    # Print summary statistics
        print(f"{fluor}: {len(filtered_components[fluor]['leds'])}/{len(available_leds)} LEDs, "
              f"{len(filtered_components[fluor]['ex_filters'])}/{len(available_ex_filters)} excitation filters, "
              f"{len(filtered_components[fluor]['em_filters'])}/{len(available_em_filters)} emission filters")
    
    return filtered_components
    
def nearest_n_multiples(x, n, multiple):
    # Find the nearest multiple of 5
    nearest = 5 * round(x / multiple)
    # Generate n multiples below and above the nearest multiple
    return [nearest + multiple * i for i in range(-n, n + 1)]    
    
### Filter and LED Optimization ###
if __name__ == "__main__":
    run_sim = True # Run sim or load results
    hide_plots = False # Don't show optimization results plot
    load_file = 'OptimResults\\ResultsFAMTexasRedCy5_5_04-02_11-33-10'
    
    if run_sim:
        # Specify fluorophore data files
        fluor_spectra_file = "Fluorophores\\Gentec_All.csv"
        fluor_props_file = "Fluorophores\\Fluorophore_Properties.xlsx"
        
        wavelengths, fluor_df = generate_fluor_df(fluor_spectra_file, fluor_props_file)
        
        # Select fluorophores
        fluor_selection = ["FAM","TexasRed","Cy5_5"]
        
        num_fluorophores = len(fluor_selection)
            
        fluor_conc = [100e-9] * num_fluorophores
        
        # Define candidate LEDs for optimization
        led_power = [1] * num_fluorophores
        
        available_leds_files = [os.path.join("LEDS", f) for f in os.listdir("LEDS") if os.path.isfile(os.path.join("LEDS", f))]
        available_leds = []
        
        # Add spectra from files
        for i,file_name in enumerate(available_leds_files):
            spectrum = np.genfromtxt(file_name, delimiter=",", skip_header=True)[:,1]
            available_leds.append(Component(i, wavelengths, spectrum, type="import", name=f"{os.path.splitext(os.path.basename(file_name))[0]}"))
            
        _, peak_ex_wavelengths, peak_em_wavelengths = get_fluorophore_colors(wavelengths,fluor_df, fluor_selection)    
            
        # Define candidate excitation filters
        available_ex_filters = []
        
        # Add existing filters
        filter_path = "Filters\\TriBandpassFAMROXCy5EX.csv"
        multiband_filters = split_multiband_filter(wavelengths,np.genfromtxt(filter_path,delimiter=',',skip_header=True)[:,1])
        for i,filter in enumerate(multiband_filters):
            available_ex_filters.append(Component(i, wavelengths, filter, type="import", name=f"{os.path.splitext(os.path.basename(filter_path))[0]}_{i}"))
        
        num_filters = len(available_ex_filters)
        ex_filter_wavelengths = list(itertools.chain.from_iterable([nearest_n_multiples(wl,2,5) for wl in peak_ex_wavelengths])) #[455, 495, 535, 550, 575, 590, 600, 655, 680]
        for i,center in enumerate(ex_filter_wavelengths): 
            spectrum = generate_light_spectra(wavelengths, center, 10, type="band")*0.9
            available_ex_filters.append(Component(i+num_filters, wavelengths, spectrum, type="band", name=f"Filter{center}_{10}nm"))
        
        # Add generated filters
        num_filters = len(available_ex_filters)
        for i,center in enumerate(ex_filter_wavelengths): #[470, 480, 580, 590, 655, 670, 680]
            spectrum = generate_light_spectra(wavelengths, center, 20, type="band")*0.9
            available_ex_filters.append(Component(i+num_filters, wavelengths, spectrum, type="band", name=f"Filter{center}_{20}nm"))
        
        # Define candidate emission filters
        available_em_filters = []
        
        # Add existing filters
        em_filter_wavelengths = list(itertools.chain.from_iterable([nearest_n_multiples(wl,2,5) for wl in peak_em_wavelengths]))
        for center in em_filter_wavelengths: # [455, 495, 535, 550, 575, 590, 600, 610, 660, 680]
            spectrum = generate_light_spectra(wavelengths, center, 10, type="band")*0.9
            available_em_filters.append(Component(i, wavelengths, spectrum,type="band",name=f"Filter{center}_{10}nm"))
        
        # Add generated filters
        num_filters = len(available_em_filters)
        for i,center in enumerate(em_filter_wavelengths): #[520,530,540,625,700,710]
            spectrum = generate_light_spectra(wavelengths, center, 25, type="band")*0.9
            available_em_filters.append(Component(i+num_filters, wavelengths, spectrum, type="band",name=f"Filter{center}_{25}nm"))
        
        num_filters = len(available_em_filters)
        em_filter_wavelengths = list(itertools.chain.from_iterable([nearest_n_multiples(wl,1,15) for wl in peak_em_wavelengths]))
        for i,center in enumerate(em_filter_wavelengths): #[530,540,690,700,710]
            spectrum = generate_light_spectra(wavelengths, center, 40, type="band")*0.9
            available_em_filters.append(Component(i+num_filters, wavelengths, spectrum, type="band",name=f"Filter{center}_{25}nm"))
        
        ### Pre filtering ###
        overlap_threshold_LED = 0.4
        overlap_threshold_ex = 0.3
        overlap_threshold_em = 0.2
        filtered_components = filter_components_by_overlap(wavelengths, fluor_df, fluor_selection, available_leds, 
                                        available_ex_filters, available_em_filters, overlap_threshold_LED,
                                        overlap_threshold_ex,overlap_threshold_em)
        
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
        
        result_data = [optimization_results,wavelengths,wavelengths,fluor_df,fluor_selection,fluor_conc,led_power]
        with open(save_file, 'wb') as f:
            pickle.dump(result_data, f)
            
        plot_optimization_results(optimization_results,wavelengths,fluor_df,fluor_selection,fluor_conc,led_power,hide_plots)

    else:
        with open(load_file, 'rb') as f:
            optimization_results,wavelengths,wavelengths,fluor_df,fluor_selection,fluor_conc,led_power = pickle.load(f)

        # Plot Pareto front
        plot_optimization_results(optimization_results,wavelengths,fluor_df,fluor_selection,fluor_conc,led_power,hide_plots)
