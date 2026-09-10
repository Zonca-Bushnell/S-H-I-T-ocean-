function ok = maybe_start_parallel_pool(workers)
    ok = false;
    if ~isfinite(workers) || workers <= 1
        return
    end
    try
        if license('test', 'Distrib_Computing_Toolbox')
            pool = gcp('nocreate');
            if ~isempty(pool)
                pool_type = '';
                try
                    pool_type = char(pool.Type);
                catch
                    pool_type = '';
                end
                if ~strcmpi(pool_type, 'threads') || pool.NumWorkers ~= workers
                    delete(pool);
                    pool = [];
                end
            end
            if isempty(pool)
                try
                    parpool('threads', workers);
                catch ME
                    warning('Thread parallel pool unavailable, using serial loop to avoid process-pool memory copies: %s', ME.message);
                    ok = false;
                    return
                end
            end
            ok = true;
        end
    catch ME
        warning('Parallel pool unavailable, using serial loop: %s', ME.message);
        ok = false;
    end
end
