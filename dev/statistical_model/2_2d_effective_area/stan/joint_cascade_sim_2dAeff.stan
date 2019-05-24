/**
 * Forward model for neutrino energies and arrival directions.
 * Focusing on cascade events for now, and ignoring different flavours and interaction types.
 * Adding in the 2D Aeff and spline implementation. 
 *
 * @author Francesca Capel
 * @date April 2019
 */

functions {

#include vMF.stan
#include interpolation.stan
#include energy_spectrum.stan
#include bspline_ev.stan

  /**
   * Get exposure factor from spline information and source positions.
   * Units of [m^2 yr]
   */
  vector get_exposure_factor(real T, real Emin, real alpha, vector alpha_grid, vector[] integral_grid, int Ns) {

    int K = Ns+1;
    vector[K] eps;
    print("K: ", K);
    
    for (k in 1:K) {

      eps[k] = interpolate(alpha_grid, integral_grid[k], alpha) * ((alpha-1) / Emin) * T;
      
    }

    return eps;
  }
  
  /**
   * Calculate weights from exposure integral.
   */
  vector get_exposure_weights(vector F, vector eps) {

    int K = num_elements(F);
    vector[K] weights;
    
    real normalisation = 0;
    
    for (k in 1:K) {
      normalisation += F[k] * eps[k];
    }

    for (k in 1:K) {
      weights[k] = F[k] * eps[k] / normalisation;
    }

    return weights;
  }
  

  /**
   * Convert from unit vector omega to theta of spherical coordinate system.
   * @param omega a 3D unit vector.
   */
  real omega_to_zenith(vector omega) {

    real zenith;
    
    int N = num_elements(omega);

    if (N != 3) {
      print("Error: input vector omega must be of 3 dimensions");
    }

    zenith = pi() - acos(omega[3]);
    
    return zenith;
  }

  /**
   * Calculate the expected number of detected events from each source.
   */
  real get_Nex_sim(vector F, vector eps) {

    int K = num_elements(F);
    real Nex = 0;

    for (k in 1:K) {
      Nex += F[k] * eps[k];
    }

    return Nex;
  }

}

data {

  /* sources */
  int<lower=0> Ns;
  unit_vector[3] varpi[Ns]; 
  vector[Ns] D;
  vector[Ns+1] z;

  /* energies */
  real<lower=1> alpha;
  real Emin; // GeV
  real f_E;

  /* deflection */
  real<lower=0> kappa;
  
  /* flux */
  real<lower=0> Q;
  real<lower=0> F0;

  /* Effective area */
  int Ngrid;
  vector[Ngrid] alpha_grid;
  vector[Ngrid] integral_grid[Ns+1];
  real T;
  
  int p; // spline degree
  int Lknots_x; // length of knot vector
  int Lknots_y; // length of knot vector

  vector[Lknots_x] xknots; // knot sequence - needs to be a monotonic sequence
  vector[Lknots_y] yknots; // knot sequence - needs to be a monotonic sequence
 
  matrix[Lknots_x+p-1, Lknots_y+p-1] c; // spline coefficients 
  real aeff_max;
  
}

transformed data {

  real<lower=0> Fs = 0;
  real<lower=0> FT;
  real<lower=0, upper=1> f;
  vector[Ns+1] F;
  simplex[Ns+1] w_exposure;
  real Nex;
  int N;
  real Mpc_to_m = 3.086e22;
  vector[Ns+1] eps;
  
  for (k in 1:Ns) {
    F[k] = Q / (4 * pi() * pow(D[k] * Mpc_to_m, 2));
    Fs += F[k];
  }
  F[Ns+1] = F0;

  FT = F0 + Fs;
  f = Fs / FT;

  /* N */
  eps = get_exposure_factor(T, Emin, alpha, alpha_grid, integral_grid, Ns);
  w_exposure = get_exposure_weights(F, eps);
  Nex = get_Nex_sim(F, eps);

  N = poisson_rng(Nex);

  /* Debug */
  print("F: ", F);
  print("Fs: ", Fs);
  print("f: ", f);
  print("w_exposure: ", w_exposure);
  print("N: ", N);
}

generated quantities {

  int lambda[N];
  unit_vector[3] omega;
  vector[N] Esrc;
  vector[N] E;
  vector[N] Edet;
  
  real zenith[N];
  real pdet[N];
  real accept;
  simplex[2] prob;
  unit_vector[3] event[N];
  real Nex_sim = Nex;

  real cosz;
  real log10E;
  
  
  for (i in 1:N) {

    /* Sample position */
    lambda[i] = categorical_rng(w_exposure);
    if (lambda[i] < Ns+1) {
      omega = varpi[lambda[i]];
    }
    else if (lambda[i] == Ns+1) {
      omega = sphere_rng(1);
    }
    zenith[i] = omega_to_zenith(omega);
  
    /* Rejection sample energy */
    accept = 0;
    while (accept != 1) {
      
      /* Sample energy */
      Esrc[i] = spectrum_rng( alpha, Emin * (1 + z[lambda[i]]) );
      E[i] = Esrc[i] / (1 + z[lambda[i]]);

      /* check bounds of spline */
      log10E = log10(E[i]);
      cosz = cos(zenith[i]);
      if (log10E >= 6.96) {
	log10E = 6.96;
      }
      if (cosz <= -0.8999) {
	cosz = -0.8999;
      }
      if (cosz >= 0.8999) {
	cosz = 0.8999;
      }
      
      /* Test against Aeff */
      pdet[i] = pow(10, bspline_func_2d(xknots, yknots, p, c, log10E, cosz)) / aeff_max;
      if (log10(E[i]) > 7.0) {
	pdet[i] = 0;
      }
      prob[1] = pdet[i];
      prob[2] = 1 - pdet[i];
      accept = categorical_rng(prob);
      
    }

    /* Detection effects */
    event[i] = vMF_rng(omega, kappa);  	  
 
    /* Simple normal for now  */
    /* To be replaced with something more realistic... */
    Edet[i] = normal_rng(E[i], f_E * E[i]);
    while (Edet[i] < Emin) {
      Edet[i] = normal_rng(E[i], f_E * E[i]);
    }
 
  }  

}


