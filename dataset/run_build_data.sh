# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
WORK_DIR="./battery_word_seg"
OUTPUT_DIR="${WORK_DIR}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

IMAGE_FOLDER="${WORK_DIR}/JPEGImages"
LIST_FOLDER="${WORK_DIR}/ImageSets/Segmentation"
SEMANTIC_SEG_FOLDER="${WORK_DIR}/SegmentationClassRaw"

echo "Converting PASCAL VOC 2012 dataset..."
python ./build_voc2012_data.py \
  --image_folder="${IMAGE_FOLDER}" \
  --semantic_segmentation_folder="${SEMANTIC_SEG_FOLDER}" \
  --list_folder="${LIST_FOLDER}" \
  --image_format="jpg" \
  --output_dir="${OUTPUT_DIR}"