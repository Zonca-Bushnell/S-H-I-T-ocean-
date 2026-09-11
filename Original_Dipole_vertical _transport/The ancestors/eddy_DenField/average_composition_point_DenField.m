%% interpolation

cd '/Users/Root/Output/Eddy Heat Flux'/DenField_ae_ISAS/
list = dir('**/*.mat');

deplen = 152;

s = RandStream('dsfmt19937');
randnum = randperm(s,length(list));

loopnum = ones(1,6)*34;

[x,y] = meshgrid(-4:0.1:4,-4:0.1:4);
Den_compound_all = NaN(81,81,deplen,length(loopnum));

%%
tic

nl = 1;

p = parpool("Threads",10);

for k = 1:length(loopnum)

    xl = loopnum(k);
    randnum_in = randnum(nl:nl+xl-1);
    nl = nl + xl;
    
    I_x_in = NaN(8e7,1,'single');
    I_y_in = NaN(8e7,1,'single');
    I_Den_in = NaN(8e7,deplen,1,'single');
    int = 1;

    % I_x_in = [];
    % I_y_in = [];
    % I_Den_in = [];

    for nn = 1:loopnum(k)
        % data combination
        path = [list(randnum_in(nn)).folder '/' list(randnum_in(nn)).name];
        load(path)

        len = length(I_x_all);
        I_x_in(int:int+len-1) = I_x_all;
        I_y_in(int:int+len-1) = I_y_all;
        I_Den_in(int:int+len-1,:) = I_Den_all;
        int = int + len;

        % I_x_in = cat(1,I_x_in,I_x_all);
        % I_y_in = cat(1,I_y_in,I_y_all);
        % I_Den_in = cat(1,I_Den_in,I_Den_all);


        disp(['nn = ',num2str(nn)])
    end

    temp = I_x_in.*I_y_in;
    index = isnan(temp);
    I_x_in(index) = [];
    I_y_in(index) = [];
    I_Den_in(index,:) = [];

    clear I_x_all I_y_all I_Den_all path temp index int len

    
    % composition
    dd = 1.2;
    Den_compound_temp = NaN(length(x),length(y),deplen);
    parfor i = 1:81
        % disp(['i = ',num2str(i)])
        for j = 1:81
            index = I_x_in >= x(i,j)-dd & I_x_in <= x(i,j)+dd & ...
                I_y_in >= y(i,j)-dd & I_y_in <= y(i,j)+dd;
            Den_compound_temp(i,j,:) = mean(I_Den_in(index,:),1,'omitnan');
        end
    end
    % delete(p)    
            
    Den_compound_all(:,:,:,k) = Den_compound_temp;
    
    disp(['k = ',num2str(k)])

    clear i j index
end

delete(p)

cd '/Users/Root/Output/Eddy Heat Flux'
filename = 'Argo04_data_DenField_ae_North_twosat_ISAS_7Sample_mean1.2.mat';
save(filename,"Den_compound_all")

toc
