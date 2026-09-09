function ok = maybe_start_parallel_pool(workers)
    ok = false;
    if ~isfinite(workers) || workers <= 1
        return
    end
    try
        if license('test', 'Distrib_Computing_Toolbox')
            pool = gcp('nocreate');
            if isempty(pool)
                parpool('threads', workers);
            end
            ok = true;
        end
    catch ME
        warning('Parallel pool unavailable, using serial sensitivity loop: %s', ME.message);
        ok = false;
    end
end
