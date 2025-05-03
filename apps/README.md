
## 3D Slicer
```
monailabel start_server --app apps/radiology --studies data/LIDC --conf models segmentation_lidc
```

## OHIF
```
monailabel start_server --app apps/radiology --studies http://127.0.0.1:8042/dicom-web --conf models segmentation_lidc
```

# Orthanc

```
sudo service orthanc stop
sudo service orthanc restart
```