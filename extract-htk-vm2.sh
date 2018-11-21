# Use OpenSMILE 2.1.0 only

if [ $# -lt 1 ]; then
  echo "USAGE: extract-htk.sh <INPUT FILE>"
  exit 1
fi

filename=$(basename "$1")
dirname=$(dirname "$1")
extension="${filename##*.}"
basename="${filename%.*}"

FEATURE_NAME=eGeMAPS #med # arbitrary - just a name
INPUT=$1
# CONFIG_FILE=/vagrant/MED_2s_100ms_htk.conf
CONFIG_FILE=./config/gemaps/eGeMAPSv01a.conf
OUTPUT_DIR=$dirname/VCMtemp/
OPENSMILE=~/tools/opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract

mkdir -p $OUTPUT_DIR
file=$INPUT
id=`basename $file`
echo "Extracting features for $id ..."
id=${id%.wav}

# Use OpenSMILE 2.3.0
# LD_LIBRARY_PATH=/home/vagrant/usr/local/lib \
LD_LIBRARY_PATH=/usr/local/lib 

# echo $OPENSMILE -C $CONFIG_FILE -I $file -O $OUTPUT_DIR/${id}.htk -logfile extract-htk.log # \

    $OPENSMILE \
    -C $CONFIG_FILE \
    -I $file \
    -htkoutput $OUTPUT_DIR/${id}.htk \
    -logfile extract-htk.log  \
     >& /dev/null

echo "DONE!"
