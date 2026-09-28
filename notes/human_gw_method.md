Human GW workflow/detectability analyis (somehwat complete?) following moore [arXiv:1408.0740v2](https://arxiv.org/abs/1408.0740)

1. Get psi4 at theta,phi for (2,2), (2,1), & (2,-2), (2, -1)
   add together, use spin weighted spherical harmonic with proper weight (-2 i think?)
2. Compute fft of psi4(theta,phi) with tukey window with good parameter choice(0.1?) and take mode average
   |Psi4_av| = sqrt((FFT(Re(psi4)) + FFT(Im(psi4))) / 2), polarization average
3. Orientation average
      1/4pi integral | Psi4_av|(theta,phi) dOmega other option is RMS sqrt(1/4pi integral (|psi4_av|^2(theta,phi)dOmega))
4. h_c = 2 f |h_tilde| = 2 f |psi4_av| / omega^2 (Moore paper!)
5. Compute h_n from S_n, h_n  = sqrt(f) sqrt(S_n) (moore paper) for detector,
   for source, h_c = sqrt(f) sqrt(S_h) source amplitude
   following moore
6. Rescale our h_c for different masses at some distance, cosmology. Plot vs f/f_orb, along with h_n plot.
7. Calculate detectability horizon for given SNR threshold.

Follow-up requests retained from the Anvil plot note:
- Investigate radial extrapolation if the extraction data support it.
- Add a temporal comparison without changing the direct-Psi4 off baseline
  (implemented September 21; model limitations remain explicit below).

The luminosity-distance plot and consolidated three-rescaling measured-signal
chart from that note are implemented. The attempted cached-strain combined
time-off/on pair was withdrawn: it changed the established off spectrum.
Individual FFI-strain tail diagnostics remain a separate experiment. Currently
A1 is conditional; none of these diagnostics validates a long-lived tail.
The restore reminder is resolved by keeping this document on both machines.

September 21 update: the replacement combined time-on comparison stays in
Psi4. Differentiate the fitted complex strain model analytically, blend this
Psi4 tail into measured (2,2), then run the same FFT/average/conversion as above.
The original off arrays are verified identical. The new flux-driven mass
proxy is optional (`TAIL_MASS_SOURCE`); A1 remains conditional and A2/A3 have
no supported waveform continuation. ET and the A-only view follow todo.txt.
