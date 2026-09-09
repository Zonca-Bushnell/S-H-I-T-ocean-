function use_gpu = should_use_gpu(compute_device)
    use_gpu = false;
    if strcmp(compute_device, 'cpu')
        fprintf('Compute device: CPU requested.\n');
        return
    end
    try
        n_gpu = gpuDeviceCount;
        if n_gpu > 0
            gpuDevice(1);
            use_gpu = true;
            fprintf('Compute device: GPU Cressman enabled on device 1.\n');
        end
    catch ME
        if strcmp(compute_device, 'gpu')
            warning('GPU requested but unavailable; falling back to CPU: %s', ME.message);
        else
            fprintf('Compute device: GPU unavailable, using CPU Cressman.\n');
        end
        use_gpu = false;
    end
end
