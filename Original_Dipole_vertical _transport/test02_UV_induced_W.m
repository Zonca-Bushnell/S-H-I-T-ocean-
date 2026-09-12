%%
load('/Users/Root/Output/Eddy Heat Flux/Argo02_data_point_ae_North_twosat.mat','E_radius')
% path = '/Users/Root/Data/AVISO_Eddy/META3.2_DT_twosat/META3.2_DT_twosat_Cyclonic_long_19930101_20220209.nc';
% radius_eddy = ncread(path,'speed_radius');
% lat_eddy = ncread(path,'latitude');
% radius_eddy = radius_eddy(lat_eddy > 0);

% load('/Users/Root/Output/Eddy Heat Flux/Argo04_data_DenField_ae_North_twosat_ISAS_7Sample.mat')

%% load Depth
% ISAS
path_test = '/Users/Root/Data/Argo_Data/ISAS_Argo/field/2004/ISAS20_ARGO_20040615_fld_TEMP.nc';
depth = double(ncread(path_test,'depth'));
depth = depth(1:152);
depth(1) = 0;
clear path_test

% % WOA
% path_test = '/Users/Root/Data/WOA/Self_WOA2023_PotentialDensity/PDen1000_13.mat';
% data = load(path_test);
% depth = double(data.depth);
% clear path_test data

%%
eddy_DenField_in = mean(Den_compound_all,4,'omitnan');

Depth1 = (0:25:2000)';
[x,y,dep] = meshgrid(-4:0.1:4,-4:0.1:4,depth);
[xv,yv,depv] = meshgrid(-4:0.1:4,-4:0.1:4,Depth1);

eddy_DenField =  interp3(x,y,dep,eddy_DenField_in,xv,yv,depv,'linear');

% Density initial data
Den_compound_smooth = NaN(size(eddy_DenField));
for i = 1:81
    temp = eddy_DenField(:,:,i);
    temp = ndnanfilter(temp,'rectwin',[4 4]);
    Den_compound_smooth(:,:,i) = temp;
end

% Den_compound_smooth = eddy_DenField;
% Den_compound_smooth = ndnanfilter(Den_compound_smooth,'rectwin',[4 4 0]);

clear x y xv yv dep depv temp

% grid distance

% Ld
% load('/Users/Root/Data/Rossby Radius/baroclinic Rossby radius/OSU/the_first_baroclinic_rossby_radius.mat')

% composition grid distance
radius_eddy = median(E_radius,'omitmissing');
% radius_eddy = mean(Ld_interp(LA == 30.375),'omitnan');

grid_dis = 8*radius_eddy/80;

clear LA LO Ld_interp i

%% isopycnal slope

% zonal slope
dzdx = NaN(size(Den_compound_smooth));

for i = 1:81
    for j = 2:81 - 1

        den2 = squeeze(Den_compound_smooth(i,j+1,:));
        den1 = squeeze(Den_compound_smooth(i,j-1,:));

        % vertical monotonically decreasing
        temp = diff(den1);
        temp = cat(1,temp,1);
        index = find(temp < 0);
        if ~isempty(index)
            if index > 79
                index = 81;
                den1(index) = den1(index-1) + (den1(index-1) - den1(index-2));
            else
                den1(index) = den1(index+1) - (den1(index+2) - den1(index+1));
            end
        end

        temp = diff(den2);
        temp = cat(1,temp,1);
        index = find(temp < 0);
        if ~isempty(index)
            if index > 79
                index = 81;
                den2(index) = den2(index-1) + (den2(index-1) - den2(index-2));
            else
                den2(index) = den2(index+1) - (den2(index+2) - den2(index+1));
            end
        end
        
        % interpolation
        F2 = griddedInterpolant(den2,-1*Depth1,'linear','none');
        F1 = griddedInterpolant(den1,-1*Depth1,'linear','none');
        
        den0 = squeeze(Den_compound_smooth(i,j,:));

        z2 = F2(den0);
        z1 = F1(den0);

        dzdx(i,j,:) = 0.5*(z2 - z1)/grid_dis;

    end
end

dzdx(:,1,:) = dzdx(:,2,:);
dzdx(:,end,:) = dzdx(:,end-1,:);

% meridional slope
dzdy = NaN(size(Den_compound_smooth));

for i = 2:81 - 1
    for j = 1:81

        den2 = squeeze(Den_compound_smooth(i+1,j,:));
        den1 = squeeze(Den_compound_smooth(i-1,j,:));

        % vertical monotonically decreasing
        temp = diff(den1);
        temp = cat(1,temp,1);
        index = find(temp < 0);
        if ~isempty(index)
            if index > 79
                index = 81;
                den1(index) = den1(index-1) + (den1(index-1) - den1(index-2));
            else
                den1(index) = den1(index+1) - (den1(index+2) - den1(index+1));
            end
        end

        temp = diff(den2);
        temp = cat(1,temp,1);
        index = find(temp < 0);
        if ~isempty(index)
            if index > 79
                index = 81;
                den2(index) = den2(index-1) + (den2(index-1) - den2(index-2));
            else
                den2(index) = den2(index+1) - (den2(index+2) - den2(index+1));
            end
        end
   
        % interpolation
        F2 = griddedInterpolant(den2,-1*Depth1,'linear','none');
        F1 = griddedInterpolant(den1,-1*Depth1,'linear','none');
        
        den0 = squeeze(Den_compound_smooth(i,j,:));

        z2 = F2(den0);
        z1 = F1(den0);

        dzdy(i,j,:) = 0.5*(z2 - z1)/grid_dis;

    end
end

dzdy(1,:,:) = dzdy(2,:,:);
dzdy(end,:,:) = dzdy(end-1,:,:);

for i = 1:81
    temp = dzdx(:,:,i);
    temp = ndnanfilter(temp,'rectwin',[2 2]);
    dzdx(:,:,i) = temp;
end

for i = 1:81
    temp = dzdy(:,:,i);
    temp = ndnanfilter(temp,'rectwin',[2 2]);
    dzdy(:,:,i) = temp;
end

clear F1 F2 i j index temp den1 den2 z1 z2

%% UV induced W
load('/Users/Root/Output/Eddy Heat Flux/rebuild_W/UV_induced_W/AE_North_ThermalWind_UV_smooth1.mat')

W_uis = U_thw.*dzdx;
W_vis = V_thw.*dzdy;
W_is = W_uis + W_vis;

%% save data

filename = 'CE_North_BackgroundDen_W_smooth2.mat';
save(filename,'dzdx','dzdy','W_is','W_uis','W_vis','Depth1','grid_dis')

