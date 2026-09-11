%% Temperature and salinity data
cd '/Users/Root/Data'/WOA/WOA2023/seasonal/
list = dir('**/*.nc');
list1 = struct2table(list);
list_name = list1.name;
list_name = char(list_name);

clear list1

%% Calculate Density

tic

namebox = [13,14,15,16];

parpool("Threads",10)

for k = 1:4

    path_S = [list(k).folder '/' list(k).name];
    path_T = [list(k+4).folder '/' list(k+4).name];
    
    lon = ncread(path_S,'lon');
    lat = ncread(path_S,'lat');
    salt = ncread(path_S,'s_an');
    temp = ncread(path_T,'t_an');

    [lat_in,lon_in,salt,temp] = recenter(lat,lon,salt,temp,'center',180);

    depth = ncread(path_S,'depth');
    dpos = depth < 2200;
    depth = depth(dpos);

    salt = salt(:,:,dpos);
    temp = temp(:,:,dpos);

    
    ilen = size(temp,1);
    
    SA = NaN(size(salt));
    parfor j = 1:size(salt,2)
        pres = gsw_p_from_z(-1*depth,lat_in(j));
        for i = 1:ilen
            SP = squeeze(salt(i,j,:));
            if sum(isnan(SP)) < 85
                SA(i,j,:) = gsw_SA_from_SP(SP,pres,lon_in(i),lat_in(j));
            end
        end
    end
    
    CT = NaN(size(temp));
    parfor j = 1:size(temp,2)
        pres = gsw_p_from_z(-1*depth,lat_in(j));
        for i = 1:ilen
            t = squeeze(temp(i,j,:));
            if sum(isnan(t)) < 85
                CT(i,j,:) = gsw_CT_from_t(squeeze(SA(i,j,:)),t,pres);
            end
        end
    end
    
    Den = NaN(size(temp));
    parfor j = 1:size(temp,2)
        for i = 1:ilen
            SA_in = squeeze(SA(i,j,:));
            CT_in = squeeze(CT(i,j,:));
            if sum(isnan(SA_in)) < 85
                Den(i,j,:) = gsw_rho(SA_in,CT_in,1000); % Reference Depth
            end
        end
    end
    
    clear i j t ilen temp salt pres

    filename = ['/Users/Root/Data/WOA/Self_WOA2023_PotentialDensity' '/PDen1000_' ...
        num2str(namebox(k))];
    save(filename,"Den","lat_in","lon_in","depth")

end

toc

%%
data_inuse = Den(:,:,47);

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
clim([1031.5 1032.5])

clear p h ans

