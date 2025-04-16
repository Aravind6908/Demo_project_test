import zipfile
import os

zipf = zipfile.ZipFile('app_bundle.zip', 'w', zipfile.ZIP_DEFLATED)
for root, dirs, files in os.walk('.'):
    for file in files:
        if file != 'app_bundle.zip':
            zipf.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file), '.'))
zipf.close()
