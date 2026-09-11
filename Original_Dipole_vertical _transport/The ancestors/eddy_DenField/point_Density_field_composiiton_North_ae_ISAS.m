%% data preparation
% fname_eddy = '/Users/Root/Data/AVISO_Eddy/META3.2_DT_twosat/META3.2_DT_twosat_Anticyclonic_long_19930101_20220209.nc';
% lat_eddy = ncread(fname_eddy,'latitude');
% lon_eddy = ncread(fname_eddy,'longitude');
% time_eddy   = ncread(fname_eddy,'time');
% radius_eddy = ncread(fname_eddy,'speed_radius');
load('/Users/Root/Data/AVISO_Eddy/EddyData_ae_taken_SEVENdays_twosat.mat')
lat_eddy = lat_eddy_new;
lon_eddy = lon_eddy_new;
time_eddy   = time_eddy_new;
radius_eddy = radius_eddy_new;

clear lat_eddy_new lon_eddy_new time_eddy_new radius_eddy_new

%%

timeday = double(time_eddy) + datenum('1950-01-01'); %#ok<DATNM>
t_vec = datevec(timeday);

clear time_eddy fname_eddy timeday

%% time selection
t_pos = find(t_vec(:,1) < 2021 & t_vec(:,1) > 2003);

lon_eddy = lon_eddy(t_pos);
lat_eddy = lat_eddy(t_pos);

radius_eddy = radius_eddy(t_pos);

t_vec = t_vec(t_pos,:);
t_year = t_vec(:,1);
t_year_uniuqe = unique(t_year);
t_ym = t_vec(:,1)*100 + t_vec(:,2);
t_ym_uniuqe = unique(t_ym);

clear t_pos t_vec_month t_vec_year

%% North and South
typeddy = lat_eddy > 0;

lon_eddy = lon_eddy(typeddy);
lat_eddy = lat_eddy(typeddy);
radius_eddy = radius_eddy(typeddy);
t_year = t_year(typeddy);
t_ym = t_ym(typeddy);

clear typeddy

%% Temperature and salinity data
cd '/Users/Root/Data/Argo_Data'/Self_ISAS_Argo_PotentialDensity/
list = dir('**/*.mat');
list1 = struct2table(list);
list_name = list1.name;
list_name = char(list_name);
list_time = list_name(:,10:15);
list_time = str2num(list_time); %#ok<ST2NM>

clear list1 list_name

% Depth1 = (0:25:2000)';
% Depth1(1) = 1;
% lat_interp = -75:0.125:75;
% lon_interp = 0:0.125:359.5;

yearmonth = 200401;
pos = list_time == yearmonth;
path = [list(pos).folder '/' list(pos).name];
load(path)

% [x,y,z] = meshgrid(lat_in,lon_in,depth);
% [xv,yv,zv] = meshgrid(lat_interp,lon_interp,Depth1);
% Den_interp = interp3(x,y,z,Den,xv,yv,zv);
Den_interp = Den(:,:,1:152);
lon_interp = lon_in;
lat_interp = lat_in;
Depth1 = depth(1:152);

clear path pos

%% begin composition!

tic

p = parpool("Threads",10);

for k = 1:length(t_ym_uniuqe)

    I = t_ym == t_ym_uniuqe(k);
    lon_eddy1 = lon_eddy(I);
    lat_eddy1 = lat_eddy(I);
    radius_eddy1 = radius_eddy(I);
    t_ym1 = t_ym(I);

    yearmonth = t_ym_uniuqe(k);
    pos = list_time == yearmonth;
    path = [list(pos).folder '/' list(pos).name];
    data = load(path);
    Den = data.Den;
    Den_interp = Den(:,:,1:152);

    nl = 1;
    lim = 0;

    I_x_all = cell(length(radius_eddy1)*400*12,1);
    I_y_all = cell(length(radius_eddy1)*400*12,1);
    I_Den_all = cell(length(radius_eddy1)*400*12,1);
    E_lat = cell(length(radius_eddy1)*400*12,1);
    E_lon = cell(length(radius_eddy1)*400*12,1);


    parfor i = 1:length(lon_eddy1)
    
        
        %----------cut gradSST data basing on eddy boundary-----------

        if lon_eddy1(i) > 0 && lon_eddy1(i) < 359.5 && ...
                abs(lat_eddy1(i)) < 75
            
            lat_c = lat_eddy1(i);
            lon_c = lon_eddy1(i);
        
            lon_min1 = lon_c - 10;lon_max1 = lon_c + 10;
            lat_min1 = lat_c - 10;lat_max1 = lat_c + 10;
    
            latp = lat_interp >= lat_min1 & lat_interp <= lat_max1;
            lonp = lon_interp >= lon_min1 & lon_interp <= lon_max1;
            lon_interp_in = lon_interp(lonp);
            lat_interp_in = lat_interp(latp);
            
            Den_in = Den_interp(lonp,latp,:);
            Den_in = permute(Den_in,[2 1 3]);
    
    
        %----------- extracting data basing on eddy boundary----------
            [LN,LA] = meshgrid(lon_interp_in,lat_interp_in);
            LN_inuse = LN;
            LA_inuse = LA;
            [dim1,dim2,~] = size(Den_in);

            arclen = distance(repmat(lat_c,dim1,dim2),repmat(lon_c,dim1,dim2),LA,LN);
            dis = arclen*3.1416/180*6378000;
            in_edge = dis > 6*radius_eddy1(i);

            LN(in_edge) = NaN;
            LA(in_edge) = NaN;

            if sum(isnan(LA),[1 2]) < dim1*dim2 - 36

                LN_temp = extract_eddy_data(LA,LN,LN_inuse);
                LA_temp = extract_eddy_data(LA,LN,LA_inuse);
    
                Den_temp = extract_eddy_data(LA,LN,Den_in);
                [dim1,dim2,~] = size(Den_temp);

                I_LN = reshape(LN_temp,[dim1*dim2 1]);
                I_LA = reshape(LA_temp,[dim1*dim2 1]);
                I_Den = reshape(Den_temp,[dim1*dim2 152 1]);

                % x-y coordinate
                arclen = distance(lat_c,I_LN,lat_c,lon_c);
                I_x = arclen*pi/180*6371000/radius_eddy1(i);
                temp = I_LN < lon_c;
                I_x(temp) = -1*I_x(temp);

                arclen = distance(I_LA,lon_c,lat_c,lon_c);
                I_y = arclen*pi/180*6371000/radius_eddy1(i);
                temp = I_LA < lat_c;
                I_y(temp) = -1*I_y(temp);

                I_x_all{i} = I_x;
                I_y_all{i} = I_y;
                I_Den_all{i} = I_Den;
                E_lat{i} = lat_c.*ones(length(I_x),1);
                E_lon{i} = lon_c.*ones(length(I_x),1);
                
            end
        end

    end


    id = cellfun('length',I_x_all);
    id = sum(id,2);

    I_x_all(id == 0) = [];
    I_y_all(id == 0) = [];
    I_Den_all(id == 0) = [];
    E_lat(id == 0) = [];
    E_lon(id == 0) = [];

    % cell to mat
    id = cellfun('length',I_x_all);
    I_x_all_temp = NaN(sum(id),1,'single');
    I_y_all_temp = NaN(sum(id),1,'single');
    I_Den_all_temp = NaN(sum(id),152,'single');
    E_lat_temp = NaN(sum(id),1,'single');
    E_lon_temp = NaN(sum(id),1,'single');

    int = 1;
    for nn = 1:length(I_x_all)
        len = id(nn);
        I_x_all_temp(int:int+len-1) = I_x_all{nn};
        I_y_all_temp(int:int+len-1) = I_y_all{nn};
        I_Den_all_temp(int:int+len-1,:) = I_Den_all{nn};
        E_lat_temp(int:int+len-1) = E_lat{nn};
        E_lon_temp(int:int+len-1) = E_lon{nn};
        int = int + len;
    end
    
    I_Den_all = I_Den_all_temp;
    I_x_all = I_x_all_temp;
    I_y_all = I_y_all_temp;
    E_lat = E_lat_temp;
    E_lon = E_lon_temp;

    clear I_x_all_temp I_y_all_temp I_Den_all_temp E_lon_temp E_lat_temp

    temp = isnan(I_Den_all);
    temp = sum(temp,2);
    temp(temp < 150) = 0;
    temp(temp > 150) = 1;
    temp = logical(temp);

    I_x_all(temp) = [];
    I_y_all(temp) = [];
    I_Den_all(temp,:) = [];
    E_lat(temp) = [];
    E_lon(temp) = [];

    filename = ['/Users/Root/Output/Eddy Heat Flux/DenField_ae_ISAS' '/' 'eddy_DenField',num2str(k)];
    save(filename,'I_x_all','I_y_all','I_Den_all','E_lat','E_lon')
 
    disp(k)
end
    

delete(p)
clear p

toc

%% function for extracting data within eddies

function data_extract = extract_eddy_data(LA,LN,data_origin)

    la1 = find(LA == min(min(LA)));                     
    [row1,~] = ind2sub(size(LA),la1(1));
    la2 = find(LA == max(max(LA)));
    [row2,~] = ind2sub(size(LA),la2(1));
    
    if row1 < row2
        edge = data_origin(row1:row2,:,:);
    else
        edge = data_origin(row2:row1,:,:);
    end

    ln1 = find(LN == min(min(LN)));
    [~,col1] = ind2sub(size(LN),ln1(1));
    ln2 = find(LN == max(max(LN)));
    [~,col2] = ind2sub(size(LN),ln2(1));

    if col1 < col2
        edge = edge(:,col1:col2,:);
    else
        edge = edge(:,col2:col1,:);
    end

    data_extract = edge;

end

