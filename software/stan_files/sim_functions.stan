#include interpolation.stan

/**
 * Get exposure factor from spline information and source positions.
 * Units of [m^2 yr]
 */
vector get_exposure_factor(real T, real Emin, real alpha, vector alpha_grid, vector[] integral_grid, int Ns) {

  int K = Ns+1;
  vector[K] eps;
    
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
real get_Nex(vector F, vector eps) {
  
  int K = num_elements(F);
  real Nex = 0;
  
  for (k in 1:K) {
    Nex += F[k] * eps[k];
  }
  
  return Nex;
}
