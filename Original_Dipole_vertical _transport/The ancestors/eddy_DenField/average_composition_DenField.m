%% interpolation

eddy_DenField_in = mean(eddy_DenField_monthly,4,'omitnan');

Depth1 = (0:25:2000)';
[x,y,dep] = meshgrid(linspace(-4,4,51),linspace(-4,4,51),Depth1);
[xv,yv,depv] = meshgrid(-4:0.1:4,-4:0.1:4,Depth1);

eddy_DenField =  interp3(x,y,dep,eddy_DenField_in,xv,yv,depv,'spline');


clear x y xv yv dep depv


