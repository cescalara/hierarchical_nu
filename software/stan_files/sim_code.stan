functions
{
#include utils.stan
#include vMF.stan
#include interpolation.stan
#include sim_functions.stan
real CascadesEffAreaHist(real value_0,real value_1)
{
real hist_array[50,10] = {{ 0.68716122, 0.80817492, 0.91932813, 0.9974366 , 1.03773818, 1.05343648,
   1.04858595, 1.04034061, 1.04054923, 1.04766316},
 { 0.80161819, 0.96739474, 1.12671833, 1.24052094, 1.30244796, 1.31844301,
   1.31382792, 1.30320484, 1.2956829 , 1.31245069},
 { 0.91962692, 1.14941919, 1.37421579, 1.52362822, 1.60376476, 1.62314537,
   1.61689301, 1.6059428 , 1.60156558, 1.62941481},
 { 1.04092977, 1.33707601, 1.627235  , 1.84170115, 1.95315021, 1.98356724,
   1.97570325, 1.9444556 , 1.94672725, 1.9932663 },
 { 1.14946332, 1.52642022, 1.92117022, 2.19336752, 2.3487031 , 2.39100385,
   2.36624627, 2.33760544, 2.33385198, 2.40303507},
 { 1.2456346 , 1.72519479, 2.22156849, 2.59716982, 2.79299169, 2.83119144,
   2.80526436, 2.7595231 , 2.75665855, 2.84381776},
 { 1.34043309, 1.91672475, 2.54745071, 3.0057239 , 3.26967682, 3.34173575,
   3.2847843 , 3.23789207, 3.23319092, 3.35768151},
 { 1.4042922 , 2.09056001, 2.85344237, 3.45833125, 3.80350657, 3.87624382,
   3.83648923, 3.74758644, 3.76416387, 3.89210122},
 { 1.46710894, 2.25680639, 3.16968692, 3.91687091, 4.35745595, 4.48349956,
   4.43306067, 4.33006325, 4.32710889, 4.51551216},
 { 1.49709354, 2.40754145, 3.48747078, 4.3974427 , 4.91996863, 5.10505897,
   5.08755857, 4.95799486, 4.94506596, 5.15759823},
 { 1.51619689, 2.52262852, 3.77528776, 4.85829817, 5.5401895 , 5.78350492,
   5.71362851, 5.61450792, 5.6152032 , 5.83537588},
 { 1.51982591, 2.59984544, 4.05158171, 5.3323175 , 6.13827915, 6.47122256,
   6.46316776, 6.30193672, 6.29172321, 6.56699198},
 { 1.50066877, 2.66094674, 4.27604101, 5.7408859 , 6.76272856, 7.1780342 ,
   7.17684515, 7.06547077, 7.04570088, 7.32608529},
 { 1.47733067, 2.70715978, 4.46655299, 6.14896173, 7.35003581, 7.91522476,
   7.98805705, 7.84145814, 7.79259887, 8.10878618},
 { 1.42909215, 2.70419846, 4.57069355, 6.48969686, 7.89608343, 8.62503123,
   8.73900362, 8.64000293, 8.57784177, 8.94539947},
 { 1.37434444, 2.67862253, 4.68464479, 6.81885365, 8.46934971, 9.34624114,
   9.56178926, 9.45408265, 9.41486761, 9.77282515},
 { 1.32700417, 2.6273235 , 4.70504017, 7.0557024 , 8.97308916,10.04221352,
  10.37124321,10.2928445 ,10.24243359,10.69869325},
 { 1.26658087, 2.54622376, 4.69677548, 7.23998494, 9.36454956,10.67108171,
  11.13081531,11.08300199,11.09813672,11.52672759},
 { 1.21753964, 2.47016756, 4.6426744 , 7.3296714 , 9.74654133,11.31605778,
  11.94883471,11.96803088,11.97329963,12.42635725},
 { 1.16476003, 2.3742978 , 4.57773204, 7.36895408,10.07726885,11.92307849,
  12.78392469,12.85704126,12.89444256,13.31767849},
 { 1.12293494, 2.27279095, 4.44170789, 7.35475876,10.29839451,12.51523922,
  13.54870687,13.8256517 ,13.83412908,14.33101471},
 { 1.08504436, 2.17321756, 4.27587745, 7.25214724,10.4857762 ,13.0042069 ,
  14.37474843,14.73549847,14.79898268,15.2015481 },
 { 1.05572999, 2.07159447, 4.12122538, 7.12568165,10.58606981,13.55333618,
  15.28135245,15.78183072,15.8595764 ,16.17901734},
 { 1.04222328, 1.97851935, 3.90534359, 6.96995785,10.69164887,14.0527671 ,
  16.15956352,16.9514571 ,16.90970719,17.13155344},
 { 1.04037506, 1.8988744 , 3.71980652, 6.77287672,10.68383153,14.56143073,
  17.11732055,18.21134647,18.19903284,18.08456867},
 { 1.04658375, 1.81546819, 3.49661248, 6.51593502,10.7347932 ,15.08605591,
  18.26240498,19.60043713,19.52325178,19.18868135},
 { 1.08553937, 1.75390494, 3.30214334, 6.27326859,10.70940707,15.62340143,
  19.46298217,21.2708268 ,20.97543287,20.33566816},
 { 1.14119755, 1.7052893 , 3.12198076, 6.00717239,10.71905766,16.18935085,
  20.78801857,23.06779155,22.78019695,21.48892342},
 { 1.21958655, 1.66535549, 2.94247808, 5.78485128,10.64420628,16.80156825,
  22.2475041 ,25.30513527,24.89709256,22.82751385},
 { 1.33560871, 1.6495563 , 2.76819788, 5.55793699,10.57980358,17.35415052,
  24.07273351,28.02244633,27.40558285,24.40138758},
 { 1.50102854, 1.64455302, 2.63372854, 5.35504859,10.56683741,18.01959653,
  25.90327328,31.11248314,30.31160174,26.16061086},
 { 0.3663096 , 0.3851484 , 0.60302809, 1.22603277, 2.45546533, 4.23700492,
   6.312594  , 7.5744099 , 7.34139307, 6.27010591},
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        },
 { 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,
   0.        , 0.        , 0.        , 0.        }};
real hist_edge_0[51] = {3.16227766e+04,3.80189396e+04,4.57088190e+04,5.49540874e+04,
 6.60693448e+04,7.94328235e+04,9.54992586e+04,1.14815362e+05,
 1.38038426e+05,1.65958691e+05,1.99526231e+05,2.39883292e+05,
 2.88403150e+05,3.46736850e+05,4.16869383e+05,5.01187234e+05,
 6.02559586e+05,7.24435960e+05,8.70963590e+05,1.04712855e+06,
 1.25892541e+06,1.51356125e+06,1.81970086e+06,2.18776162e+06,
 2.63026799e+06,3.16227766e+06,3.80189396e+06,4.57088190e+06,
 5.49540874e+06,6.60693448e+06,7.94328235e+06,9.54992586e+06,
 1.14815362e+07,1.38038426e+07,1.65958691e+07,1.99526231e+07,
 2.39883292e+07,2.88403150e+07,3.46736850e+07,4.16869383e+07,
 5.01187234e+07,6.02559586e+07,7.24435960e+07,8.70963590e+07,
 1.04712855e+08,1.25892541e+08,1.51356125e+08,1.81970086e+08,
 2.18776162e+08,2.63026799e+08,3.16227766e+08};
real hist_edge_1[11] = {-1. ,-0.8,-0.6,-0.4,-0.2, 0. , 0.2, 0.4, 0.6, 0.8, 1. };
return hist_array[binary_search(value_0, hist_edge_0)][binary_search(value_1, hist_edge_1)];
}
real spectrum_rng(real alpha,real e_low,real e_up)
{
real uni_sample;
real norm;
norm = ((1-alpha)/((e_up^(1-alpha))-(e_low^(1-alpha))));
uni_sample = uniform_rng(0, 1);
return ((((uni_sample*(1-alpha))/norm)+(e_low^(1-alpha)))^(1/(1-alpha)));
}
real flux_conv(real alpha,real e_low,real e_up)
{
real f1;
real f2;
if(alpha == 1.0)
{
f1 = (log(e_up)-log(e_low));
}
else
{
f1 = ((1/(1-alpha))*((e_up^(1-alpha))-(e_low^(1-alpha))));
}
if(alpha == 2.0)
{
f2 = (log(e_up)-log(e_low));
}
else
{
f2 = ((1/(2-alpha))*((e_up^(2-alpha))-(e_low^(2-alpha))));
}
return (f1/f2);
}
vector CascadesAngularResolution_rng(real true_energy,vector true_dir)
{
vector[6] CascadesAngularResolutionPolyCoeffs = [-4.84839608e-01, 3.59082699e+00, 4.39765349e+01,-4.86964043e+02,
  1.50499694e+03,-1.48474342e+03]';
return vMF_rng(true_dir, eval_poly1d(log10(truncate_value(true_energy, 100.0, 100000000.0)),CascadesAngularResolutionPolyCoeffs));
}
real c_energy_res_mix_rng(vector means,vector sigmas,vector weights)
{
int index;
index = categorical_rng(weights);
return lognormal_rng(means[index], sigmas[index]);
}
real CascadeEnergyResolution_rng(real true_energy)
{
real CascadesEnergyResolutionMuPolyCoeffs[3,4] = {{ 8.48311816e-02,-1.40745871e+00, 8.39735975e+00,-1.29122823e+01},
 { 2.21533176e-02,-3.66019621e-01, 2.96495763e+00,-3.60384905e+00},
 { 2.32768756e-03,-4.24291670e-02, 1.26012779e+00,-5.56780566e-01}};
real CascadesEnergyResolutionSdPolyCoeffs[3,4] = {{-4.14191929e-03, 7.53090020e-02,-4.31439499e-01, 8.45584789e-01},
 { 1.31648640e-03,-2.44148959e-02, 1.55383236e-01,-3.02974554e-01},
 {-4.15523836e-04, 7.44664372e-03,-4.42253583e-02, 9.70242677e-02}};
real mu_e_res[3];
real sigma_e_res[3];
vector[3] weights;
for (i in 1:3)
{
weights[i] = 1.0/3;
}
for (i in 1:3)
{
mu_e_res[i] = eval_poly1d(log10(truncate_value(true_energy, 1000.0, 10000000.0)), to_vector(CascadesEnergyResolutionMuPolyCoeffs[i]));
sigma_e_res[i] = eval_poly1d(log10(truncate_value(true_energy, 1000.0, 10000000.0)), to_vector(CascadesEnergyResolutionSdPolyCoeffs[i]));
}
return c_energy_res_mix_rng(to_vector(log(mu_e_res)), to_vector(sigma_e_res), weights);
}
real CascadesAngularResolution(real true_energy,vector true_dir,vector reco_dir)
{
vector[6] CascadesAngularResolutionPolyCoeffs = [-4.84839608e-01, 3.59082699e+00, 4.39765349e+01,-4.86964043e+02,
  1.50499694e+03,-1.48474342e+03]';
return vMF_lpdf(reco_dir | true_dir, eval_poly1d(log10(truncate_value(true_energy, 100.0, 100000000.0)),CascadesAngularResolutionPolyCoeffs));
}
real c_energy_res_mix(real x,vector means,vector sigmas,vector weights)
{
vector[3] result;
for (i in 1:3)
{
result[i] = (log(weights)[i]+lognormal_lpdf(x | means[i], sigmas[i]));
}
return log_sum_exp(result);
}
real CascadeEnergyResolution(real true_energy,real reco_energy)
{
real CascadesEnergyResolutionMuPolyCoeffs[3,4] = {{ 8.48311816e-02,-1.40745871e+00, 8.39735975e+00,-1.29122823e+01},
 { 2.21533176e-02,-3.66019621e-01, 2.96495763e+00,-3.60384905e+00},
 { 2.32768756e-03,-4.24291670e-02, 1.26012779e+00,-5.56780566e-01}};
real CascadesEnergyResolutionSdPolyCoeffs[3,4] = {{-4.14191929e-03, 7.53090020e-02,-4.31439499e-01, 8.45584789e-01},
 { 1.31648640e-03,-2.44148959e-02, 1.55383236e-01,-3.02974554e-01},
 {-4.15523836e-04, 7.44664372e-03,-4.42253583e-02, 9.70242677e-02}};
real mu_e_res[3];
real sigma_e_res[3];
vector[3] weights;
for (i in 1:3)
{
weights[i] = 1.0/3;
}
for (i in 1:3)
{
mu_e_res[i] = eval_poly1d(log10(truncate_value(true_energy, 1000.0, 10000000.0)), to_vector(CascadesEnergyResolutionMuPolyCoeffs[i]));
sigma_e_res[i] = eval_poly1d(log10(truncate_value(true_energy, 1000.0, 10000000.0)), to_vector(CascadesEnergyResolutionSdPolyCoeffs[i]));
}
return c_energy_res_mix(log10(reco_energy), to_vector(log(mu_e_res)), to_vector(sigma_e_res), weights);
}
real CascadesEffectiveArea(real true_energy,vector true_dir)
{
return CascadesEffAreaHist(true_energy, cos(pi() - acos(true_dir[3])));
}
}
data
{
int Ns;
unit_vector[3] varpi[Ns];
vector[Ns] D;
vector[Ns+1] z;
real alpha;
real Edet_min;
real Esrc_min;
real Esrc_max;
real L;
real F_diff;
int Ngrid;
vector[Ngrid] alpha_grid;
vector[Ngrid] integral_grid[Ns+1];
real aeff_max;
real v_lim;
real T;
}
transformed data
{
vector[Ns+1] F;
simplex[Ns+1] w_exposure;
vector[Ns+1] eps;
int track_type;
int cascade_type;
real Ftot;
real Fs;
real f;
real Nex;
int N;
track_type = 0;
cascade_type = 1;
Fs = 0.0;
for (k in 1:Ns)
{
F[k] = L/ (4 * pi() * pow(D[k] * 3.086e+22, 2));
F[k]*=flux_conv(alpha, Esrc_min, Esrc_max);
Fs += F[k];
}
F[Ns+1] = F_diff;
Ftot = (Fs+F_diff);
f = Fs/Ftot;
print("f: ", f);
eps = get_exposure_factor(alpha, alpha_grid, integral_grid, T, Ns);
Nex = get_Nex(F, eps);
w_exposure = get_exposure_weights(F, eps);
N = poisson_rng(Nex);
print(w_exposure);
print(Ngrid);
print(Nex);
print(N);
}
generated quantities
{
int Lambda[N];
unit_vector[3] omega;
vector[N] Esrc;
vector[N] E;
vector[N] Edet;
real cosz[N];
real Pdet[N];
int accept;
int detected;
int ntrials;
simplex[2] prob;
unit_vector[3] event[N];
real Nex_sim;
vector[N] event_type;
Nex_sim = Nex;
for (i in 1:N)
{
Lambda[i] = categorical_rng(w_exposure);
accept = 0;
detected = 0;
ntrials = 0;
while((accept!=1))
{
if(Lambda[i] <= Ns)
{
omega = varpi[Lambda[i]];
}
else if(Lambda[i] == (Ns+1))
{
omega = sphere_lim_rng(1, v_lim);
}
cosz[i] = cos(omega_to_zenith(omega));
if(Lambda[i] <= (Ns+1))
{
Esrc[i] = spectrum_rng(alpha, Esrc_min, Esrc_max);
E[i] = (Esrc[i]/(1+z[Lambda[i]]));
}
Pdet[i] = (CascadesEffectiveArea(E[i], omega)/aeff_max);
Edet[i] = (10^CascadeEnergyResolution_rng(E[i]));
prob[1] = Pdet[i];
prob[2] = (1-Pdet[i]);
ntrials += 1;
if(ntrials< 1000000)
{
detected = categorical_rng(prob);
if((Edet[i] >= Edet_min) && ((detected==1)))
{
accept = 1;
}
}
else
{
accept = 1;
print("problem component: ", Lambda[i]);
;
}
}
event[i] = CascadesAngularResolution_rng(E[i], omega);
event_type[i] = cascade_type;
}
}
