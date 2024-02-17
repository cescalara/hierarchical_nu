/**
 * Stan functions for interpolation.
 * 
 * @author Francesca Capel
 * @date February 2019
 */

/**
 * Interpolate x from a given set of x and y values.
 * Prints warning if x is outside of the interpolation range.
 */
real interpolate(vector x_values, vector y_values, real x) {

  real x_left;
  real y_left;
  real x_right;
  real y_right;
  real dydx;
  
  int Nx = num_elements(x_values);
  real xmin = x_values[1];
  real xmax = x_values[Nx];
  int i = 1;

  if (x > xmax || x < xmin) {

    /*
    print("Warning, x is outside of interpolation range!");
    print("Returning edge values.");
    print("x:", x);
    print("xmax", xmax);
    */
    
    if(x > xmax) {
      return y_values[Nx];
    }
    else if (x < xmin) {
      return y_values[1];
    }
  }
    
  if( x >= x_values[Nx - 1] ) {
    i = Nx - 1;
  }
  else {
    while( x > x_values[i + 1] ) { i = i+1; }
  }

  x_left = x_values[i];
  y_left = y_values[i];
  x_right = x_values[i + 1];
  y_right = y_values[i + 1];
  
  dydx = (y_right - y_left) / (x_right - x_left);
    
  return y_left + dydx * (x - x_left);
}

real interpolate_log_y(vector x_values, vector log_y_values, real x) {

  real x_left;
  real y_left;
  real x_right;
  real y_right;
  real dydx;
  
  int Nx = num_elements(x_values);
  real xmin = x_values[1];
  real xmax = x_values[Nx];
  int i = 1;

  if (x > xmax || x < xmin) {

    /*
    print("Warning, x is outside of interpolation range!");
    print("Returning edge values.");
    print("x:", x);
    print("xmax", xmax);
    */
    
    if(x > xmax) {
      return exp(log_y_values[Nx]);
    }
    else if (x < xmin) {
      return exp(log_y_values[1]);
    }
  }
    
  if( x >= x_values[Nx - 1] ) {
    i = Nx - 1;
  }
  else {
    while( x > x_values[i + 1] ) { i = i+1; }
  }

  x_left = x_values[i];
  y_left = log_y_values[i];
  x_right = x_values[i + 1];
  y_right = log_y_values[i + 1];
  
  dydx = (y_right - y_left) / (x_right - x_left);
    
  return exp(y_left + dydx * (x - x_left));
}


real interp2d(real x, real y, array[] real xp, array[] real yp, array[,] real fp) {
  int idx_y = binary_search(y, yp);
  int idx_yp1 = idx_y + 1;
  //safeguard against y values outside the defined range
  // interpolate will take care of the same issue in x direction
  if (idx_y == 0) {
    // return result from lowest slice
    return interpolate(to_vector(xp), to_vector(fp[:, 1]), x);
  }
  else if (idx_y >= size(yp)) {
    return interpolate(to_vector(xp), to_vector(fp[:, size(yp)]), x);
  }
  real y_vals_low = interpolate(to_vector(xp), to_vector(fp[:, idx_y]), x);
  real y_vals_high = interpolate(to_vector(xp), to_vector(fp[:, idx_yp1]), x);
  real val = interpolate(to_vector(yp[idx_y:idx_yp1]), [y_vals_low, y_vals_high]', y);
  return val;
}