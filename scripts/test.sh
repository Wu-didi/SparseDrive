bash ./tools/dist_test.sh \
    projects/configs/sparsedrive_small_stage2.py \
    /mnt/home/wudi/code_v3/SparseDrive/work_dirs/sparsedrive_small_stage2/iter_93760.pth \
    1 \
    --deterministic \
    --eval bbox
    # --result_file ./work_dirs/sparsedrive_small_stage2/results.pkl