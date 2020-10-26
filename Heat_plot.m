x_intersections_BCE(x_intersections_BCE==0) = nan;
y_intersections_BCE(y_intersections_BCE==0) = nan;
complex_wave_mach_BCE(complex_wave_mach_BCE==0) = nan;
x_intersections_DFG(x_intersections_DFG==0) = nan;
y_intersections_DFG(y_intersections_DFG==0) = nan;
complex_wave_mach_DFG(complex_wave_mach_DFG==0) = nan;
x_intersections_HIJ(x_intersections_HIJ==0) = nan;
y_intersections_HIJ(y_intersections_HIJ==0) = nan;
complex_wave_mach_HIJ(complex_wave_mach_HIJ==0) = nan;

x_intersections = [x_intersections_BCE, x_intersections_DFG, x_intersections_HIJ];
y_intersections = [y_intersections_BCE, y_intersections_DFG, y_intersections_HIJ];
mach_dist = [complex_wave_mach_BCE, complex_wave_mach_DFG, complex_wave_mach_HIJ];
figure
patch(x_intersections, y_intersections, mach_dist)
colorbar