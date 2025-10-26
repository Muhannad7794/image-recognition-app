## data upload
To upload data using the ingestion API from EI, navigate to the main dir were that data lives and run:

for dir in */; do \
    label=$(basename "$dir"); \
    echo "Uploading '$label' with automatic 80/20 splitting..."; \
    edge-impulse-uploader --label "$label" --category split "$dir"* ; \
done