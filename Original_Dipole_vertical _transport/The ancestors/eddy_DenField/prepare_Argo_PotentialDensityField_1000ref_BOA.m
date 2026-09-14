%% Temperature and salinity data
cd '/Users/Root/Data/Argo_Data'/BOA_Argo/MAT/
list = dir('**/*.mat');
list1 = struct2table(list);
list_name = list1.name;
list_name = char(list_name);
list_time1 = list_name(:,10:13);
list_time1 = list_time1(1:225,:);
list_time2 = list_name(:,15:16);
list_time2 = list_time2(1:225,:);

list_time1 = str2num(list_time1); %#ok<ST2NM>
list_time2 = str2num(list_time2); %#ok<ST2NM>

list_time = list_time1*100 + list_time2;

list_time_unique = unique(list_time);

clear list1 list_name list_time1 list_time2

path_test = '/Users/Root/Data/Argo_Data/BOA_Argo/MAT/BOA_Argo_2004_01.mat';
load(path_test)

clear temp* salt* mld*

%% Calculate Density

tic

parpool('Threads',10)

for k = 1:length(list_time_unique)

    pos = find(list_time == list_time_unique(k));
    
    path = [list(pos(1)).folder '/' list(pos(1)).name];
    
    load(path)

    lon_in = lon(:,1);
    lat_in = lat(1,:);
    lat_in = lat_in';
    
    ilen = size(temp,1);
    
    SA = NaN(size(salt));
    parfor j = 1:size(salt,2)
        for i = 1:ilen
            SP = squeeze(salt(i,j,:));
            if sum(isnan(SP)) < 85
                SA(i,j,:) = gsw_SA_from_SP(SP,pres,lon_in(i),lat_in(j));
            end
        end
    end
    
    CT = NaN(size(temp));
    parfor j = 1:size(temp,2)
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
                Den(i,j,:) = gsw_rho(SA_in,CT_in,1000);
            end
        end
    end
    
    clear i j t ilen temp* salt* mld*

    filename = ['/Users/Root/Data/Argo_Data/Self_BOA_Argo_PotentialDensity' '/PDen1000_' ...
        num2str(list_time_unique(k))];
    save(filename,"Den","lat_in","lon_in","pres")

end

toc

%%
data_inuse = Den(:,:,41);

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
% clim([1031.5 1033])

clear p h ans

