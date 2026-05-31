#!/bin/bash
# Download pretrained model checkpoints from Dropbox
# Run from tracks/industrial-infineon/

set -e

mkdir -p models/decoder models/encoder

echo "Downloading decoder..."
wget -q -O /tmp/procseq_models.zip "https://www.dropbox.com/scl/fo/6oqw3d0tkymlrd05nxtl9/AMRL--KrlBNRk1W2ENqdmtg?rlkey=wlfuwq3nfoq33z24k4qu6etbh&dl=1"

echo "Extracting..."
unzip -o /tmp/procseq_models.zip -d /tmp/procseq_models
cp -r /tmp/procseq_models/procseq/decoder/* models/decoder/
cp -r /tmp/procseq_models/procseq/encoder/* models/encoder/
rm -rf /tmp/procseq_models /tmp/procseq_models.zip

echo "Done. Models in models/decoder/ and models/encoder/"
