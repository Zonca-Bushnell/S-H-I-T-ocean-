%% data preparation
fname_eddy = '/Users/Root/Data/AVISO_Eddy/META3.2_DT_twosat/META3.2_DT_twosat_Anticyclonic_long_19930101_20220209.nc';
lat_eddy = ncread(fname_eddy,'latitude');
lon_eddy = ncread(fname_eddy,'longitude');
time_eddy   = ncread(fname_eddy,'time');
radius_eddy = ncread(fname_eddy,'speed_radius');
track_eddy = ncread(fname_eddy,'track');

%% each eddy only takes SEVEN shapshot

track_unique = unique(track_eddy);

lat_eddy_cell = cell(length(track_unique),1);
lon_eddy_cell = cell(length(track_unique),1);
time_eddy_cell = cell(length(track_unique),1);
radius_eddy_cell = cell(length(track_unique),1);

tic

p = parpool("Threads",10);
parfor i = 1:length(track_unique)
    pos = find(track_eddy == track_unique(i));
    lat_eddy_in = lat_eddy(pos);
    lon_eddy_in = lon_eddy(pos);
    time_eddy_in = time_eddy(pos);
    radius_eddy_in = radius_eddy(pos);

    lifelen = length(pos);
    dis = fix(lifelen/7);
    index = 1:dis:dis*7;

    lat_eddy_cell{i} = lat_eddy_in(index);
    lon_eddy_cell{i} = lon_eddy_in(index);
    time_eddy_cell{i} = time_eddy_in(index);
    radius_eddy_cell{i} = radius_eddy_in(index);
end
delete(p);clear p

lat_eddy_new = NaN(length(track_unique)*7,1);
lon_eddy_new = NaN(length(track_unique)*7,1);
time_eddy_new = NaN(length(track_unique)*7,1);
radius_eddy_new = NaN(length(track_unique)*7,1);

int = 1;
for i = 1:length(track_unique)
    lat_eddy_new(int:int+6) = lat_eddy_cell{i};
    lon_eddy_new(int:int+6) = lon_eddy_cell{i};
    time_eddy_new(int:int+6) = time_eddy_cell{i};
    radius_eddy_new(int:int+6) = radius_eddy_cell{i};
    int = int + 7;
end

lat_eddy = lat_eddy_new;
lon_eddy = lon_eddy_new;
time_eddy = time_eddy_new;
radius_eddy = radius_eddy_new;

clear i int pos radius_eddy_in time_eddy_in lon_eddy_in lat_eddy_in ...
    lat_eddy_new lon_eddy_new time_eddy_new radius_eddy_new dis lifelen index ...
    lat_eddy_cell lon_eddy_cell time_eddy_cell radius_eddy_cell track_eddy track_unique

toc

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
t_month = t_vec(:,2);

t_ym = t_vec(:,1)*100 + t_vec(:,2);
t_ym_uniuqe = unique(t_ym);

t_month(t_month >= 1 & t_month <= 3) = 13;
t_month(t_month >= 4 & t_month <= 6) = 14;
t_month(t_month >= 7 & t_month <= 9) = 15;
t_month(t_month >= 10 & t_month <= 12) = 16;

clear t_pos t_vec_month t_vec_year t_vec

%% North and South
typeddy = lat_eddy > 0;

lon_eddy = lon_eddy(typeddy);
lat_eddy = lat_eddy(typeddy);
radius_eddy = radius_eddy(typeddy);
t_ym = t_ym(typeddy);
t_month = t_month(typeddy);

clear typeddy fname_eddy

%% Temperature and salinity data
cd '/Users/Root/Data'/WOA/Self_WOA2023_PotentialDensity/
list = dir('**/*.mat');

for ii = 1:4
    path = [list(ii).folder '/' list(ii).name];
    load(path)
    f = list(ii).name;
    f = f(1:11);
    com = append(f,' = Den;');
    eval(com)
end

clear ii f com path Den list

lon_interp = lon_in;
lat_interp = lat_in;
Depth1 = depth;
deplen = length(Depth1);

%% begin composition!

tic

for k = 1:length(t_ym_uniuqe)

    I = t_ym == t_ym_uniuqe(k);
    lon_eddy1 = lon_eddy(I);
    lat_eddy1 = lat_eddy(I);
    radius_eddy1 = radius_eddy(I);
    t_ym1 = t_ym(I);
    t_month1 = t_month(I);

    com = append('Den_interp = ','PDen1000_',num2str(t_month1(1)),';');
    eval(com)

    I_x_all = cell(length(radius_eddy1)*400*12,1);
    I_y_all = cell(length(radius_eddy1)*400*12,1);
    I_Den_all = cell(length(radius_eddy1)*400*12,1);

    p = parpool("Threads",10);
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
                I_Den = reshape(Den_temp,[dim1*dim2 deplen 1]);

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
                
            end
        end

    end

    delete(p)
    clear p

    id = cellfun('length',I_x_all);
    id = sum(id,2);

    I_x_all(id == 0) = [];
    I_y_all(id == 0) = [];
    I_Den_all(id == 0) = [];

    % cell to mat
    id = cellfun('length',I_x_all);
    I_x_all_temp = NaN(sum(id),1,'single');
    I_y_all_temp = NaN(sum(id),1,'single');
    I_Den_all_temp = NaN(sum(id),deplen,'single');

    int = 1;
    for nn = 1:length(I_x_all)
        len = id(nn);
        I_x_all_temp(int:int+len-1) = I_x_all{nn};
        I_y_all_temp(int:int+len-1) = I_y_all{nn};
        I_Den_all_temp(int:int+len-1,:) = I_Den_all{nn};
        int = int + len;
    end
    
    I_Den_all = I_Den_all_temp;
    I_x_all = I_x_all_temp;
    I_y_all = I_y_all_temp;

    clear I_x_all_temp I_y_all_temp I_Den_all_temp id nn len int

    temp = isnan(I_Den_all);
    temp = sum(temp,2);
    temp(temp < deplen-1) = 0;
    temp(temp > deplen-1) = 1;
    temp = logical(temp);

    I_x_all(temp) = [];
    I_y_all(temp) = [];
    I_Den_all(temp,:) = [];

    filename = ['/Users/Root/Output/Eddy Heat Flux/DenField_ae_WOA' '/' 'eddy_DenField',num2str(k)];
    save(filename,'I_x_all','I_y_all','I_Den_all')
 
    disp(k)
end
    

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

