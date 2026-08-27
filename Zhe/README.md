# Copernicus ACC raw-download workflow

This workflow avoids the old FTP service and avoids `subset` transfer overhead by downloading original Copernicus Marine files first, then cropping locally.

## Environment

Use the existing mamba environment:

```powershell
mamba activate copernicus_downloading
python -c "import copernicusmarine, xarray, netCDF4, dask, cftime; print('OK')"
```

If credentials are not already saved, log in once:

```powershell
copernicusmarine login
```

The checked environment already contains the needed packages. To recreate it later:

```powershell
mamba env create -f environment-copernicus_downloading.yml
```

Environment variables are also supported:

```powershell
$env:COPERNICUSMARINE_SERVICE_USERNAME = "your_username"
$env:COPERNICUSMARINE_SERVICE_PASSWORD = "your_password"
```

## Create the file list

```powershell
python download_acc_raw.py
```

This creates:

```text
manifests\selected_files.txt
```

## Download raw files

```powershell
python download_acc_raw.py --download --reuse-file-list
```

Raw files are written under:

```text
E:\DATA\Copernicus_Data\ACC_raw
```

## Crop locally

```powershell
python crop_acc_raw.py
```

The cropped NetCDF output is written under:

```text
E:\DATA\Copernicus_Data\ACC
```

## Sample test

For a small test window, use `--sample` consistently:

```powershell
python download_acc_raw.py --sample
python download_acc_raw.py --sample --download --reuse-file-list
python crop_acc_raw.py --sample
```
