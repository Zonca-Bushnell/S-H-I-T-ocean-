%% load Density and C
% Density
load('/Users/Root/Output/Eddy Heat Flux/Argo03_compound_ce_North_res0.6_median.mat','Den_compound','Depth1')
% eddy radius
load('/Users/Root/Output/Eddy Heat Flux/Argo02_data_point_ce_North_twosat.mat','E_radius','E_mspeed')
% path = '/Users/Root/Data/AVISO_Eddy/META3.2_DT_twosat/META3.2_DT_twosat_Cyclonic_long_19930101_20220209.nc';
% radius_eddy = ncread(path,'speed_radius');
% lat_eddy = ncread(path,'latitude');
% radius_eddy = radius_eddy(lat_eddy > 0);

radius_eddy = mean(E_radius,'omitmissing');
c0 = median(E_mspeed,'omitmissing');

clear path

%% Density initial data
Den_compound_smooth = NaN(size(Den_compound));
for i = 1:81
    temp = Den_compound(:,:,i);
    temp = ndnanfilter(temp,'rectwin',[8 8]);
    Den_compound_smooth(:,:,i) = temp;
end

% Den_compound_smooth = ndnanfilter(Den_compound,'rectwin',[15 15 10]);

% grid distance

% Ld
% load('/Users/Root/Data/Rossby Radius/baroclinic Rossby radius/OSU/the_first_baroclinic_rossby_radius.mat')
% radius_eddy = mean(Ld_interp(LA == 30.375),'omitnan');

% composition grid distance

grid_dis = 8*radius_eddy/80;

clear LA LO Ld_interp i temp

% isopycnal slope

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

%% save data

filename = 'CE_North_dzdt_W_smooth15.mat';
save(filename,'dzdx','dzdy','W_dzdt','Depth1','c0','grid_dis')

%% decomposition function

function [temp_di,temp_mono] = di_mono_separate(temp_in)
    
    len = length(temp_in);
    
    [Lx,Ly] = meshgrid(linspace(-1,1,len),linspace(-1,1,len));
    [~,rho] = cart2pol(Lx,Ly);
    dis = abs(rho);
    
    temp_mono = NaN(len,len);
    
    d = max(dis(:))/len;
    
    for i = 1:len
        edge = dis >= d*(i-1) & dis < d*i;
        temp_mono(edge) = mean(temp_in(edge),'omitnan');
    end
    temp_mono = ndnanfilter(temp_mono,'rectwin',[1 1]);

    temp_di = temp_in - temp_mono;

end


