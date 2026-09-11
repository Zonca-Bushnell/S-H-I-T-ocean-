%% grid distance
load('/Users/Root/Output/Eddy Heat Flux/Argo02_data_point_ce_North_twosat.mat','Depth1','E_lat')

path = '/Users/Root/Data/AVISO_Eddy/META3.2_DT_twosat/META3.2_DT_twosat_Cyclonic_long_19930101_20220209.nc';
radius_eddy = ncread(path,'speed_radius');
lat_eddy = ncread(path,'latitude');
radius_eddy = radius_eddy(lat_eddy > 0);

radius_eddy = mean(radius_eddy,'omitmissing');
grid_dis = 8*radius_eddy/80;

% eddy composition data
load('/Users/Root/Output/Eddy Heat Flux/Argo03_compound_ce_North_res0.6_median.mat', ...
    'Den_compound','U1000_compound','V1000_compound','W1000_compound')

% background Density Field
load('/Users/Root/Output/Eddy Heat Flux/Argo04_data_DenField_ce_North_twosat_ISAS_7Sample.mat')

% eddy zonal moving speed
load('/Users/Root/Output/Eddy Heat Flux/eddy_features/eddy_moving_speed/Twosat_CE_eddy_zonal_moving_speed.mat')
moving_speed_zonal = moving_speed_zonal(lat_eddy > 0);
c0 = abs(median(moving_speed_zonal,'omitmissing'));

%% thermal wind
% gravity
g = 9.81;
lat_eddy = median(E_lat,'omitmissing');
f = 2*7.292*1e-5*sind(lat_eddy);

Den_compound_smooth = NaN(size(Den_compound));
for i = 1:81
    temp = Den_compound(:,:,i);
    temp = ndnanfilter(temp,'rectwin',[2 2]); % smooth 2
    Den_compound_smooth(:,:,i) = temp;
end

% thermal wind

% velocity from thermal wind equation
U_thw = NaN(size(Den_compound_smooth));
V_thw = NaN(size(Den_compound_smooth));

% velocity at 1000m depth
U_thw(:,:,41) = U1000_compound;
V_thw(:,:,41) = V1000_compound;

for n = 41:-1:2

    Den_inuse = Den_compound_smooth(:,:,n);
    gradx = NaN(size(U1000_compound));
    grady = NaN(size(U1000_compound));
    
    % x-direction rho gradient
    for j = 2:size(gradx,1)-1
        gradx(:,j) = 0.5*(Den_inuse(:,j+1) - Den_inuse(:,j-1))/grid_dis;
    end
    gradx(:,1) = (Den_inuse(:,2) - Den_inuse(:,1))/grid_dis;
    gradx(:,end) = (Den_inuse(:,end) - Den_inuse(:,end-1))/grid_dis;
    
    % y-direction rho gradient
    for i = 2:size(grady,1)-1
        grady(i,:) = 0.5*(Den_inuse(i+1,:) - Den_inuse(i-1,:))/grid_dis;
    end
    grady(1,:) = (Den_inuse(2,:) - Den_inuse(1,:))/grid_dis;
    grady(end,:) = (Den_inuse(end,:) - Den_inuse(end-1,:))/grid_dis;
    
    % vertical veloctiy gradient
    dudz = g/f*grady/mean(Den_compound(:),'omitnan');
    dvdz = -1*g/f*gradx/mean(Den_compound(:),'omitnan');
    
    U_thw(:,:,n-1) = dudz*25 + U_thw(:,:,n);
    V_thw(:,:,n-1) = dvdz*25 + V_thw(:,:,n);

end

for n = 42:81

    Den_inuse = Den_compound(:,:,n-1);
    gradx = NaN(size(U1000_compound));
    grady = NaN(size(U1000_compound));
    
    % x-direction rho gradient
    for j = 2:size(gradx,1)-1
        gradx(:,j) = 0.5*(Den_inuse(:,j+1) - Den_inuse(:,j-1))/grid_dis;
    end
    gradx(:,1) = (Den_inuse(:,2) - Den_inuse(:,1))/grid_dis;
    gradx(:,end) = (Den_inuse(:,end) - Den_inuse(:,end-1))/grid_dis;
    
    % y-direction rho gradient
    for i = 2:size(grady,1)-1
        grady(i,:) = 0.5*(Den_inuse(i+1,:) - Den_inuse(i-1,:))/grid_dis;
    end
    grady(1,:) = (Den_inuse(2,:) - Den_inuse(1,:))/grid_dis;
    grady(end,:) = (Den_inuse(end,:) - Den_inuse(end-1,:))/grid_dis;
    
    % vertical veloctiy gradient
    dudz = g/f*grady/mean(Den_compound(:),'omitnan');
    dvdz = -1*g/f*gradx/mean(Den_compound(:),'omitnan');
    
    U_thw(:,:,n) = dudz*-25 + U_thw(:,:,n-1);
    V_thw(:,:,n) = dvdz*-25 + V_thw(:,:,n-1);

end


% for i = 1:81
%     temp = U_thw(:,:,i);
%     temp = ndnanfilter(temp,'rectwin',[1 1]);
%     U_thw(:,:,i) = temp;
% end
% 
% for i = 1:81
%     temp = V_thw(:,:,i);
%     temp = ndnanfilter(temp,'rectwin',[1 1]);
%     V_thw(:,:,i) = temp;
% end

clear i j n gradx grady dudz dvdz temp

%% UV induced W
% ISAS
path_test = '/Users/Root/Data/Argo_Data/ISAS_Argo/field/2004/ISAS20_ARGO_20040615_fld_TEMP.nc';
depth = double(ncread(path_test,'depth'));
depth = depth(1:152);
depth(1) = 0;

eddy_DenField_in = mean(Den_compound_all,4,'omitnan');

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

clear x y xv yv dep depv temp path_test eddy_DenField_in

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

clear F1 F2 i j index temp den1 den2 z1 z2 den0

W_uis = U_thw.*dzdx;
W_vis = V_thw.*dzdy;
W_is = W_uis + W_vis;

%% dzdt induced W

Den_compound_smooth = NaN(size(Den_compound));
for i = 1:81
    temp = Den_compound(:,:,i);
    temp = ndnanfilter(temp,'rectwin',[6 6]);
    Den_compound_smooth(:,:,i) = temp;
end

% Den_compound_smooth = ndnanfilter(Den_compound,'rectwin',[11 11 5]);

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


% for i = 1:81
%     temp1 = dzdx(:,:,i);
%     temp2 = dzdy(:,:,i);
% 
%     dzdx(:,:,i) = ndnanfilter(temp1,'rectwin',[4 4]);
%     dzdy(:,:,i) = ndnanfilter(temp2,'rectwin',[4 4]);
% 
% end

clear F1 F2 i j index temp den1 den2 z1 z2 temp1 temp2

% dzdt based on observed C

W_dzdt = c0*dzdx;

%% rebuild W

W = W_dzdt + W_is;

%% save data

filename = 'test_AE_North_rebuild_W.mat';
save(filename,'W_is','W_dzdt','W','Depth1')

clear filename

