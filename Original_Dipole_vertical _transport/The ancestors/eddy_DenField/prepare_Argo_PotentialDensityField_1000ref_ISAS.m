%% Temperature and salinity data
cd '/Users/Root/Data/Argo_Data/ISAS_Argo/field/'
list = dir('**/*.nc');
list1 = struct2table(list);
list_name = list1.name;
list_name = char(list_name);
list_time = list_name(:,13:18);
list_time = str2num(list_time); %#ok<ST2NM>
list_time_unique = unique(list_time);

clear list1 list_name

path_test = '/Users/Root/Data/Argo_Data/ISAS_Argo/field/2004/ISAS20_ARGO_20040615_fld_TEMP.nc';

lon = double(ncread(path_test,'longitude'));
lat = double(ncread(path_test,'latitude'));
depth = double(ncread(path_test,'depth'));

%% Calculate Density

tic

parpool('Threads',10)

for k = 108:length(list_time_unique)

    pos = find(list_time == list_time_unique(k));
    
    path_S = [list(pos(1)).folder '/' list(pos(1)).name];
    path_T = [list(pos(2)).folder '/' list(pos(2)).name];

    TEMP = ncread(path_T,'TEMP');
    PSAL = ncread(path_S,'PSAL');
    
    [~,~,TEMP] = recenter(lat,lon,TEMP,'center',180);
    [lat_in,lon_in,PSAL] = recenter(lat,lon,PSAL,'center',180);
    
    ilen = size(PSAL,1);
    
    SA = NaN(size(PSAL));
    parfor j = 1:size(PSAL,2)
        pres = gsw_p_from_z(-1*depth,lat_in(j));
        for i = 1:ilen
            SP = squeeze(PSAL(i,j,:));
            if sum(isnan(SP)) < 85
                SA(i,j,:) = gsw_SA_from_SP(SP,pres,lon_in(i),lat_in(j));
            end
        end
    end
    
    CT = NaN(size(PSAL));
    parfor j = 1:size(PSAL,2)
        pres = gsw_p_from_z(-1*depth,lat_in(j));
        for i = 1:ilen
            t = squeeze(TEMP(i,j,:));
            if sum(isnan(t)) < 85
                CT(i,j,:) = gsw_CT_from_t(squeeze(SA(i,j,:)),t,pres);
            end
        end
    end
    
    
    Den = NaN(size(PSAL));
    parfor j = 1:size(PSAL,2)
        % pres = gsw_p_from_z(-1*depth,lat_in(j));
        for i = 1:ilen
            SA_in = squeeze(SA(i,j,:));
            CT_in = squeeze(CT(i,j,:));
            if sum(isnan(SA_in)) < 85
                Den(i,j,:) = gsw_rho(SA_in,CT_in,1000);
            end
        end
    end
    
    clear pres i j t SP ilen

    filename = ['/Users/Root/Data/Argo_Data/Self_ISAS_Argo_Density' '/PDen1000_' ...
        num2str(list_time_unique(k))];
    save(filename,"Den","lat_in","lon_in","depth")

end

toc

%%
data_inuse = Den(:,:,102);

figure
set(gcf,'position',[100 100 1000 600])
m_proj('Mercator','lon',[0 360],'lat',[-80 80]); % Miller投影
m_grid('box','on','linest','none','linewidth',3,'tickdir','out','backcolor','w','Fontsize',22,'tickstyle','dd','xtick',8,'ytick',7);
hold on
p = m_pcolor(lon_in,lat_in,data_inuse');
p.EdgeColor = 'none';
p.FaceColor = 'interp';
m_coast('patch',[.7 .7 .7],'edgecolor','none');
colormap(cmocean('balance'))
h = colorbar;
h.FontSize = 28;
clim([1031.5 1033])

clear p h ans

%%
figure
set(gcf,'position',[100 100 1000 600])
data_inuse = squeeze(Den(290,:,:));
p = pcolor(lat_in,-1*depth,data_inuse');
p.EdgeColor = 'none';
p.FaceColor = 'interp'; 
colormap(cmocean('balance'))

clear p
