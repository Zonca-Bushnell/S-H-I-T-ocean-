%% data preparation
fname_eddy = '/Users/Root/Data/AVISO_Eddy/META3.2_DT_twosat/META3.2_DT_twosat_Anticyclonic_long_19930101_20220209.nc';
lat_eddy = ncread(fname_eddy,'latitude');
lon_eddy = ncread(fname_eddy,'longitude');
time_eddy   = ncread(fname_eddy,'time');
radius_eddy = ncread(fname_eddy,'speed_radius');

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
Den_interp = Den(:,:,1:153);
lon_interp = lon_in;
lat_interp = lat_in;
Depth1 = depth(1:153);

clear path pos

%% begin composition!
d = 51;

tic

eddy_DenField_monthly = NaN(d,d,153,length(t_ym_uniuqe),'single');

% parpool("Threads",10)

for k = 1%:length(t_ym_uniuqe)

    I = t_ym == t_ym_uniuqe(k);
    lon_eddy1 = lon_eddy(I);
    lat_eddy1 = lat_eddy(I);
    radius_eddy1 = radius_eddy(I);
    t_ym1 = t_ym(I);
    
    eddy_DenField = NaN(d,d,153,length(lat_eddy1),'single');

    yearmonth = t_ym_uniuqe(k);
    pos = list_time == yearmonth;
    path = [list(pos).folder '/' list(pos).name];
    load(path);
    % Den_interp = interp3(x,y,z,Den,xv,yv,zv);
    Den_interp = Den(:,:,1:153);
 %%
    for i = 36000%1:length(lon_eddy1)
    
        
        %----------cut gradSST data basing on eddy boundary-----------
    
        if lon_eddy1(i) > 0 && lon_eddy1(i) < 359.5 && ...
                abs(lat_eddy1(i)) < 75
            
        
            lon_min1 = lon_eddy1(i) - 10;lon_max1 = lon_eddy1(i) + 10;
            lat_min1 = lat_eddy1(i) - 10;lat_max1 = lat_eddy1(i) + 10;
    
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

            arclen = distance(repmat(lat_eddy1(i),dim1,dim2),repmat(lon_eddy1(i),dim1,dim2),LA,LN);
            dis = arclen*3.1416/180*6378000;
            in_edge = dis > 4*radius_eddy1(i);

            LN(in_edge) = NaN;
            LA(in_edge) = NaN;

            if sum(isnan(LA),[1 2]) < dim1*dim2 - 36

                LN_temp = extract_eddy_data(LA,LN,LN_inuse);
                LA_temp = extract_eddy_data(LA,LN,LA_inuse);
    
                Den_temp = extract_eddy_data(LA,LN,Den_in);
                [dim1,dim2,~] = size(Den_temp);
             
            %------------------- interpolate eddy data --------------------
    
                if dim1 > 4 && dim2 > 4
                    
                    [LN_temp,LA_temp,dep_temp] = meshgrid(LN_temp(1,:),LA_temp(:,1),Depth1);
    
                    [LN_interp,LA_interp,dep_interp] = meshgrid(linspace(min(LN_temp(:)),max(LN_temp(:)),d),...
                        linspace(min(LA_temp(:)),max(LA_temp(:)),d),Depth1);    
    
                    Den_temp_interp = interp3(LN_temp,LA_temp,dep_temp,Den_temp,LN_interp,LA_interp,dep_interp);

                    eddy_DenField(:,:,:,i) = Den_temp_interp;

                end                
            end
        end

    end
    %%
    filename = ['/Users/Root/Output/Eddy Heat Flux/test_DenField' '/' 'eddy_DenField',num2str(k)];
    save(filename,'eddy_DenField')
    eddy_DenField_monthly(:,:,:,k) = median(eddy_DenField,4,'omitmissing');

    disp(k)
end
    
cd '/Users/Root/Output/Eddy Heat Flux/'
fname = 'test03_eddy_DenField_ae_North_twosat';
save(fname,'eddy_DenField_monthly')
    
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

