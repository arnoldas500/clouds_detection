################################################################################

# archive downloader / unpacker - (c) 2018 Toby Breckon, Durham University, UK

################################################################################

URL_ARCHIVE=https://collections.durham.ac.uk/downloads/r2jh343s319
ARCHIVE_DIR_LOCAL_TARGET=dataset

ARCHIVE_FILE_NAME=guo-raindrop-detection-dataset-2018.zip
ARCHIVE_DIR_NAME_UNZIPPED=guo-raindrop-detection-dataset-2018
ARCHIVE_MD5_SUM=4575a5abd4431051a2272dcd2731c63f

UNPACK_COMMAND='unzip -q'
SEMANTIC_NAME="rain drop dataset image files" # what we are downloading

################################################################################

# set this script to fail on error

set -e

# check for required commands to download and md5 check

(command -v curl | grep curl > /dev/null) ||
  (echo "Error: curl command not found, cannot download!")

(command -v md5sum | grep md5sum > /dev/null) ||
  (echo "Error: md5sum command not found, md5sum check will fail!")

################################################################################

# perform download

echo "Downloading $SEMANTIC_NAME ..."

mkdir -p $ARCHIVE_DIR_LOCAL_TARGET

ARCHIVE=./$ARCHIVE_DIR_LOCAL_TARGET/$ARCHIVE_FILE_NAME

curl --progress-bar $URL_ARCHIVE > $ARCHIVE

################################################################################

# perform md5 check and move to required local target directory

cd $ARCHIVE_DIR_LOCAL_TARGET

echo "checking the MD5 checksum for downloaded archive ..."

CHECK_SUM_CHECKPOINTS="$ARCHIVE_MD5_SUM  $ARCHIVE_FILE_NAME"

echo $CHECK_SUM_CHECKPOINTS | md5sum -c

echo "Unpacking the downloaded archive file..."

$UNPACK_COMMAND $ARCHIVE_FILE_NAME

echo "Tidying up..."

mv $ARCHIVE_DIR_NAME_UNZIPPED/* .

rm $ARCHIVE_FILE_NAME && rm -r $ARCHIVE_DIR_NAME_UNZIPPED

################################################################################

echo "... completed -> required $SEMANTIC_NAME are in $ARCHIVE_DIR_LOCAL_TARGET/"

################################################################################
