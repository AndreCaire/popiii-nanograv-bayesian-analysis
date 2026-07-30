import numpy as np
import warnings
import os
from pathlib import Path
from scipy.integrate import simpson
from enterprise.signals import parameter
from ceffyl.Ceffyl import ceffyl, signal 
from sampler_utils import setup_sampler
from astropy.cosmology import FlatLambdaCDM
import astropy.constants as const

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CEFFYL_DATA_DIR = (
    PROJECT_ROOT / "data" / "30f_fs{cp}_ceffyl"
)

CEFFYL_DATA_DIR = Path(
    os.environ.get(
        "CEFFYL_DATA_DIR",
        str(DEFAULT_CEFFYL_DATA_DIR),
    )
).expanduser()

RESULTS_DIR = PROJECT_ROOT / "results"

warnings.filterwarnings('ignore')

# Constants 
G = const.G.value
c = const.c.value
M_sun = const.M_sun.value
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# Integration grid
z_arr = np.linspace(0.01, 6.0, 100)
mc_arr = np.logspace(6, 10, 100) * M_sun 
ZZ, MM = np.meshgrid(z_arr, mc_arr)

# Cosmic distance
dl_vals = cosmo.luminosity_distance(z_arr).to('m').value
DL_GRID = np.tile(dl_vals, (len(mc_arr), 1))

# Toy Model
def toy_model_psd(freqs, Tspan, log10_R0, alpha, beta, gamma, **kwargs):
    input_shape = freqs.shape
    f_vec = freqs.flatten()
    
    # fixed parameters
    z0 = 2.0
    M_star = 1e8 * M_sun
    M_max = 1e11 * M_sun 
    
    R0 = 10**log10_R0
    
    # Merger Rate
    redshift_part = (1 + ZZ)**alpha * np.exp(-(ZZ/z0)**beta)
    mass_part = (MM/M_star)**(-gamma) * np.exp(-MM/M_max)
    rate_dlnMc = R0 * redshift_part * mass_part
    
    # Conversion dlnMc -> dMc
    rate_dMc = rate_dlnMc / MM
    
    # All together
    physics_spatial = (G * MM)**(5.0/3.0) / (np.pi**(2.0/3.0) * c**2 * DL_GRID**2)
    integrand_spatial = rate_dMc * physics_spatial
    
    # Integration
    integral_Mc = simpson(integrand_spatial, mc_arr, axis=0)
    total_hc_squared_spatial = simpson(integral_Mc, z_arr)
    
    # PSD 
    hc_sq = total_hc_squared_spatial * (f_vec**(-4.0/3.0))
    psd = hc_sq / (12 * np.pi**2 * f_vec**3)
    rho2 = psd / Tspan
    
    # Numerical Protection
    rho2[rho2 <= 0] = 1e-80
    return rho2.reshape(input_shape)

# --- Sampler Setup ---
print("Loading Ceffyl likelihood data...")

if not CEFFYL_DATA_DIR.exists():
    raise FileNotFoundError(
        "Ceffyl likelihood data directory not found: "
        f"{CEFFYL_DATA_DIR}\n"
        "Set the CEFFYL_DATA_DIR environment variable to the "
        "correct directory."
    )

ce = ceffyl(str(CEFFYL_DATA_DIR))
# --- PRIORS ---
params_toy = [
    parameter.Uniform(-60,-15)('log10_R0'), 
    parameter.Uniform(0, 6)('alpha'),       
    parameter.Uniform(0, 10)('beta'),      
    parameter.Uniform(2, 3)('gamma')        
]

sig_toy = signal(psd=toy_model_psd, params=params_toy, name='toy_model', 
                 N_freqs=ce.N_freqs, common_process=True)

ce.add_signals([sig_toy])

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sampler = setup_sampler(
    ce,
    str(RESULTS_DIR),
    ce.ln_likelihood,
    ce.ln_prior,
    resume=False,
    jump=True,
)
# Initial points
x0 = np.array([-21.3, 3.0, 5.0, 2.5])

print("\n--- Starting MCMC ---")

# Sampler interactions
sampler.sample(x0, 400000)

