
[Ubuntu Setup](https://slicer.readthedocs.io/en/latest/user_guide/getting_started.html#system-requirements)


```
sudo apt-get install libglu1-mesa libpulse-mainloop-glib0 libnss3 libasound2 qt5dxcb-plugin libsm6
```

## Push Dcom to Orthanc

```
python infers/ImportDicomFiles.py localhost 8042 ./data/LIDC_Dcom/
```